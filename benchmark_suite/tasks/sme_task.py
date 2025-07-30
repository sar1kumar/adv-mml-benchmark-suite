import os
import json
import random
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from benchmark_suite.tasks.base_task import BaseTask
from benchmark_suite.models.base_model import BaseModel
from benchmark_suite.utils.prompts.sme_prompts import create_prompt
from benchmark_suite.metrics.detection import Detection
from benchmark_suite.metrics.accuracy import Accuracy

class SMETask(BaseTask):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = config["data_path"]
        self.split = config.get("split", ["train", "val", "test"])
        self.output_dir = config.get("output_dir", "results")
        self.prompt_config = config.get("prompt_config", {})
        self.sample_size = config.get("sample_size", None)
        self.random_seed = config.get("random_seed", 42)
        self.data = {}
        
    def load_data(self) -> None:
        """Load SME dataset"""
        for split_name in self.split:
            json_file = os.path.join(self.data_path, f"SME_{split_name}.json")
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"SME data not found at {json_file}")
                
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                if self.sample_size is not None:
                    random.seed(self.random_seed)
                    if self.sample_size > len(data):
                        print(f"Warning: Requested sample size {self.sample_size} is larger than dataset size {len(data)}. Using full dataset.")
                        self.data[split_name] = data
                    else:
                        sampled_ids = random.sample(list(data.keys()), self.sample_size)
                        self.data[split_name] = {id: data[id] for id in sampled_ids}
                        print(f"Sampled {self.sample_size} examples from {split_name} split")
                else:
                    self.data[split_name] = data
                
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format SME question with context about the image using few-shot examples"""
        return create_prompt(
            question=example['question'],
            num_prompts=self.prompt_config.get('num_prompts', 8),
            method=self.prompt_config.get('method', 'random'),
            seed=self.random_seed
        )
        
    def compute_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """
        Compute metrics for SME task including answer accuracy and box IOU metrics.
        
        Args:
            predictions: List of prediction dictionaries containing model responses
            targets: List of target dictionaries containing ground truth
            
        Returns:
            Dict containing accuracy and IOU metrics
        """
        metrics = {}
        
        # Extract answers from predictions
        pred_answers = []
        for pred in predictions:
            try:
                # Handle the new JSON format
                if isinstance(pred, dict) and isinstance(pred.get("content", {}), dict):
                    content = pred["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        text = content["parts"][0].get("text", "")
                        # Extract answer from the format "Answer:\n{answer}\nExplanation:..."
                        answer_lines = [line for line in text.split('\n') if line.startswith('Answer:')]
                        if answer_lines:
                            answer = answer_lines[0].replace('Answer:', '').strip()
                            pred_answers.append(answer.lower())
                        else:
                            pred_answers.append("")
                else:
                    # Fallback to old format
                    answer = pred.get('model_prediction', '').split('\n')[0].strip()
                    pred_answers.append(answer.lower())
            except Exception as e:
                print(f"Error extracting answer from prediction: {e}")
                pred_answers.append("")
        
        # Extract ground truth answers
        target_answers = [t['ground_truth'].strip().lower() for t in targets]
        
        # Calculate answer accuracy
        accuracy_metric = Accuracy()
        metrics['answer_accuracy'] = accuracy_metric.compute(
            predictions=pred_answers,
            references=target_answers
        )
        
        # Calculate box IOU metrics using the Detection metric
        detection_metric = Detection(iou_threshold=0.5)
        iou_score = detection_metric.compute(
            predictions=predictions,  # Pass the full prediction dicts
            references=targets       # Pass the full target dicts
        )
        
        metrics['mean_iou'] = iou_score
        
        return metrics
    
    def evaluate_predictions_file(self, predictions_file: str) -> Dict[str, float]:
        """
        Evaluate metrics on an existing predictions file.
        
        Args:
            predictions_file: Path to the predictions JSON file
            
        Returns:
            Dict containing computed metrics
        """
        # Load predictions file
        with open(predictions_file, 'r') as f:
            results = json.load(f)
        
        # Extract predictions and targets
        predictions = results["predictions"]
        targets = [{
            "ground_truth": p["ground_truth"],
            "boxes": p["boxes"]
        } for p in predictions]
        
        # Compute metrics
        metrics = self.compute_metrics(
            predictions=predictions,
            targets=targets
        )
        
        print("\nMetrics for", predictions_file)
        print("Number of examples:", len(predictions))
        print("Answer Accuracy:", metrics['answer_accuracy'])
        print("Mean IOU:", metrics['mean_iou'])
        
        return metrics
        
    def evaluate(self, model: BaseModel) -> Dict[str, Any]:
        """Generate predictions for SME task and compute metrics"""
        if not self.data:
            self.load_data()
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        results = {
            "metadata": {
                "model_name": model.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "dataset": "SME",
                "splits": self.split,
                "prompt_config": self.prompt_config,
                "sample_size": self.sample_size,
                "random_seed": self.random_seed
            },
            "predictions": []
        }
        
        for split_name in self.split:
            if split_name not in self.data:
                continue
                
            split_data = self.data[split_name]
            print(f"Processing {len(split_data)} examples from {split_name} split")
            
            for example_id, example_data in split_data.items():
                image_filename = f"{example_data['imageId']}.jpg"
                image_path = os.path.join(self.data_path, image_filename)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}")
                    continue
                    
                prompt = self.format_prompt(example_data)
                prediction = model.process_image(image_path, prompt)
                model_answer = str(prediction.get("response", "")).strip()
                
                result_entry = {
                    "split": split_name,
                    "id": example_id,
                    "image_id": example_data["imageId"],
                    "image_file": image_filename,
                    "question": example_data["question"],
                    "ground_truth": example_data["answer"],
                    "explanation": example_data["explanation"],
                    "boxes": example_data["boxes"],
                    "model_prediction": model_answer,
                    "prompt": prompt,
                    "time_metrics": prediction.get("ollama_metrics", {})
                }
                results["predictions"].append(result_entry)
        
        # Compute metrics
        metrics = self.compute_metrics(
            predictions=results["predictions"],
            targets=[{
                "ground_truth": p["ground_truth"],
                "boxes": p["boxes"]
            } for p in results["predictions"]]
        )
        results["metrics"] = metrics
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_suffix = f"_sample{self.sample_size}" if self.sample_size else ""
        output_file = os.path.join(
            self.output_dir, 
            f"sme_predictions{sample_suffix}_{timestamp}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Predictions saved to {output_file}")
        
        return {
            "total_predictions": len(results["predictions"]),
            "output_file": output_file,
            **metrics
        }

    def supports_batch_processing(self) -> bool:
        """Enable batch processing for SME task."""
        return True

    def get_image_paths(self) -> List[str]:
        """Return a list of all unique local image paths required for the task."""
        if not self.data:
            self.load_data()
        
        all_image_paths = set()
        images_dir = os.path.join(self.data_path, "images")
        
        for split_name in self.split:
            if split_name not in self.data:
                continue
            split_data = self.data[split_name]
            
            for example_id, example_data in split_data.items():
                image_filename = f"{example_data['imageId']}.jpg"
                full_path = os.path.join(images_dir, image_filename)
                if os.path.exists(full_path):
                    all_image_paths.add(full_path)
        
        return list(all_image_paths)

    def prepare_batch_data(self) -> List[Dict[str, Any]]:
        """Load and structure the SME dataset for batch processing."""
        if not self.data:
            self.load_data()
        
        batch_data = []
        for split_name in self.split:
            if split_name not in self.data:
                continue
            split_data = self.data[split_name]
            
            for example_id, example_data in split_data.items():
                image_filename = f"{example_data['imageId']}.jpg"
                
                # Create prompt using the same method as realtime evaluation
                prompt = create_prompt(
                    question=example_data['question'],
                    num_prompts=self.prompt_config.get('num_prompts', 8),
                    method=self.prompt_config.get('method', 'random'),
                    seed=self.random_seed
                )
                
                batch_example = {
                    "example_id": example_id,
                    "split": split_name,
                    "image_id": example_data["imageId"],
                    "question": example_data["question"],
                    "ground_truth": example_data["answer"],
                    "boxes": example_data["boxes"],
                    "image_filename": image_filename,
                    "prompt": prompt
                }
                batch_data.append(batch_example)
        
        return batch_data

    def format_batch_request(self, example: Dict[str, Any], gcs_image_prefix: str) -> Dict[str, Any]:
        """Format a single SME example for the Vertex AI batch prediction API."""
        prompt = example["prompt"]
        gcs_image_path = f"{gcs_image_prefix}/{example['image_filename']}"
        
        return {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "file_data": {
                                    "file_uri": gcs_image_path,
                                    "mime_type": "image/jpeg"
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 1.0,
                    "maxOutputTokens": 8192,
                    "top_p": 0.95,
                    "top_k": 64
                }
            }
        }

    def evaluate_batch_results(self, batch_results_path: str, metadata_path: str) -> Dict[str, Any]:
        """Process completed batch job results and compute metrics."""
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load and parse batch results
        batch_predictions = []
        with open(batch_results_path, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    response_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    batch_predictions.append(response_text.strip())
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Warning: Could not parse batch result line: {line}. Error: {e}")
                    batch_predictions.append("")
        
        # Verify matching numbers
        if len(batch_predictions) != len(metadata):
            raise ValueError(f"Mismatch: {len(batch_predictions)} predictions vs {len(metadata)} metadata entries")
        
        # Format predictions and targets to match compute_metrics expectations
        predictions = []
        targets = []
        
        # Reconstruct the prediction and target format expected by compute_metrics
        if self.data:
            # Get original data to reconstruct boxes information
            data_lookup = {}
            for split_name in self.split:
                if split_name in self.data:
                    data_lookup.update(self.data[split_name])
            
            for i, (batch_pred, meta_item) in enumerate(zip(batch_predictions, metadata)):
                # Find the original data entry to get boxes
                example_id = meta_item.get("example_id")
                original_data = data_lookup.get(example_id, {})
                
                # Format prediction dictionary (similar to realtime evaluation format)
                pred_dict = {
                    "content": {
                        "parts": [{"text": batch_pred}]
                    },
                    "model_prediction": batch_pred,
                    "ground_truth": meta_item["ground_truth"],
                    "boxes": original_data.get("boxes", [])
                }
                predictions.append(pred_dict)
                
                # Format target dictionary
                target_dict = {
                    "ground_truth": meta_item["ground_truth"],
                    "boxes": original_data.get("boxes", [])
                }
                targets.append(target_dict)
        else:
            # Fallback if original data is not available
            for i, (batch_pred, meta_item) in enumerate(zip(batch_predictions, metadata)):
                pred_dict = {
                    "content": {
                        "parts": [{"text": batch_pred}]
                    },
                    "model_prediction": batch_pred,
                    "ground_truth": meta_item["ground_truth"],
                    "boxes": []  # Empty boxes if not available
                }
                predictions.append(pred_dict)
                
                target_dict = {
                    "ground_truth": meta_item["ground_truth"],
                    "boxes": []  # Empty boxes if not available
                }
                targets.append(target_dict)
        
        # Compute metrics using the existing compute_metrics method
        metrics = self.compute_metrics(predictions, targets)
        
        # Create detailed results
        results = {
            "metadata": {
                "model_name": "gemini-batch",
                "timestamp": datetime.now().isoformat(),
                "dataset": "SME",
                "total_examples": len(predictions),
                "batch_processing": True
            },
            "metrics": metrics,
            "summary": {
                "total_predictions": len(predictions),
                "processing_mode": "batch"
            }
        }
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"sme_batch_results_{timestamp}.json")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Batch evaluation results saved to {output_file}")
        print(f"Metrics: {metrics}")
        
        return metrics