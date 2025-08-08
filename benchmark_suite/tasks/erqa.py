import os
import json
import random
from typing import Dict, List, Any
from datetime import datetime

import pandas as pd
from benchmark_suite.tasks.base_task import BaseTask
from benchmark_suite.models.base_model import BaseModel
from benchmark_suite.metrics import BLEU, METEOR, ROUGEL, CIDEr, Accuracy

class ERQATask(BaseTask):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = config["data_path"]  # Path to erqa.parquet
        self.split = config.get("split", ["test"])
        # Get output directory from config, with fallback to default
        self.output_dir = config.get("output_dir", "results")
        if isinstance(self.output_dir, str):
            self.output_dir = os.path.join(self.output_dir, "erqa")  # Add task-specific subfolder
        self.sample_size = config.get("sample_size", None)
        self.random_seed = config.get("random_seed", 42)
        self.metrics_config = config.get("metrics", ["accuracy", "bleu", "rouge"])
        self.data = {}
        
        # Initialize metrics
        self.metric_calculators = {
            "accuracy": Accuracy(),
            "bleu": BLEU(n=4),
            "meteor": METEOR(),
            "rouge": ROUGEL(),
            "cider": CIDEr(),
        }
        
    def load_data(self) -> None:
        """Load ERQA dataset from CSV file and images folder"""
        csv_file = os.path.join(self.data_path, "output.csv")
        images_dir = os.path.join(self.data_path, "images")
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"ERQA CSV data not found at {csv_file}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"ERQA images directory not found at {images_dir}")
        
        print(f"Loading ERQA dataset from {csv_file}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Convert DataFrame to list of dicts, loading images from the images directory
            all_data = []
            for _, row in df.iterrows():
                try:
                    # Convert row to dict
                    item = row.to_dict()
                    
                    # Convert string representation of lists back to actual lists
                    if isinstance(item['image_paths'], str):
                        # Handle string representation of list
                        image_paths = eval(item['image_paths'])  # Safely evaluate string representation of list
                    else:
                        image_paths = item['image_paths']
                    
                    if isinstance(item['visual_indices'], str):
                        item['visual_indices'] = eval(item['visual_indices'])
                    
                    if isinstance(item['question_type'], str):
                        item['question_type'] = eval(item['question_type'])[0]  # Take first question type if it's a list
                    item['image_paths'] = [os.path.join(images_dir, img_path) for img_path in image_paths]
                    all_data.append(item)
                    
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            if not all_data:
                raise ValueError("No data was loaded from the CSV file")
            
            print(f"Total questions loaded: {len(all_data)}")
            
            # Group data by question type
            question_type_groups = {}
            for item in all_data:
                q_type = item["question_type"]
                if q_type not in question_type_groups:
                    question_type_groups[q_type] = []
                question_type_groups[q_type].append(item)
            
            print("\nQuestion type statistics:")
            for q_type, items in question_type_groups.items():
                print(f"{q_type}: {len(items)} questions")
            
            # Apply sampling if specified
            if self.sample_size is not None:
                random.seed(self.random_seed)
                sampled_data = []
                
                for q_type, items in question_type_groups.items():
                    if self.sample_size > len(items):
                        print(f"Warning: Requested sample size {self.sample_size} is larger than available questions ({len(items)}) for type {q_type}. Using all available examples.")
                        sampled_data.extend(items)
                    else:
                        sampled_items = random.sample(items, self.sample_size)
                        sampled_data.extend(sampled_items)
                        print(f"Sampled {self.sample_size} examples from type {q_type}")
                
                self.data = sampled_data
                print(f"\nTotal sampled questions: {len(self.data)}")
            else:
                self.data = all_data
                print(f"\nUsing all {len(self.data)} questions")
                
        except Exception as e:
            raise Exception(f"Error loading ERQA dataset: {e}")
        
        
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format ERQA question with context"""
        prompt = "You are a robotics expert. Please analyze the scene and answer the following question.\n\n"
        
        # Add question type and ID
        prompt += f"Question Type: {example['question_type']}\n"
        
        # Add the main question
        prompt += f"Question: {example['question']}\n"
        
        # Add visual indices information if available
        if 'visual_indices' in example and example['visual_indices']:
            indices = example['visual_indices']
            if isinstance(indices, list):
                if len(indices) > 1:
                    prompt += f"Focus on these parts of the images: {', '.join(map(str, indices))}\n"
                else:
                    prompt += f"Focus on this part of the image: {indices[0]}\n"
        
        # Add temporal context if multiple images
        images = example["image_paths"]
        if isinstance(images, list) and len(images) > 1:
            prompt += f"\nThis question involves analyzing a sequence of {len(images)} images showing different stages or viewpoints of the robotic scene.\n"
        
        prompt += "\nSelect the correct answer from the above options:\n"
        prompt += "Answer: "
        return prompt
        
    def compute_metrics(self, predictions: List[str], targets: List[str],
                       question_types: List[str]) -> Dict[str, float]:
        """Compute metrics overall and per question type"""
        metrics = {}
        
        # Overall metrics
        for metric_name in self.metrics_config:
            if metric_name in self.metric_calculators:
                score = self.metric_calculators[metric_name].compute(predictions, targets)
                metrics[metric_name] = score
        
        # Per question type metrics
        unique_types = set(question_types)
        for q_type in unique_types:
            type_indices = [i for i, qt in enumerate(question_types) if qt == q_type]
            type_preds = [predictions[i] for i in type_indices]
            type_targets = [targets[i] for i in type_indices]
            
            for metric_name in self.metrics_config:
                if metric_name in self.metric_calculators:
                    score = self.metric_calculators[metric_name].compute(
                        type_preds, type_targets
                    )
                    metrics[f"{q_type}_{metric_name}"] = score
        
        return metrics
        
    def evaluate(self, model: BaseModel) -> Dict[str, float]:
        """Generate predictions and compute metrics"""
        if not self.data:
            self.load_data()
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        results = {
            "metadata": {
                "model_name": model.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "dataset": "ERQA",
                "metrics_config": self.metrics_config,
                "sample_size": self.sample_size,
                "random_seed": self.random_seed
            },
            "predictions": []
        }
        
        all_predictions = []
        all_targets = []
        all_question_types = []
        
        for example in self.data:
            # Get images directly from the example
            images = example["image_paths"]
            if not isinstance(images, list):
                images = [images]
                
            prompt = self.format_prompt(example)
            
            try:
                prediction = model.process_images(images, prompt)
                model_answer = str(prediction.get("response", "")).strip()
                
                all_predictions.append(model_answer)
                all_targets.append(example["answer"])
                all_question_types.append(example["question_type"])
                
                # Store prediction details
                result_entry = {
                    "question_type": example["question_type"],
                    "question": example["question"],
                    "ground_truth": example["answer"],
                    "model_prediction": model_answer,
                    "prompt": prompt,
                    "num_images": len(images),
                    "visual_indices": example.get("visual_indices", []),
                    "time_metrics": prediction.get("ollama_metrics", {})
                }
                results["predictions"].append(result_entry)
                
            except Exception as e:
                raise e
                print(f"Error processing example {example['question']}: {e}")
                print(f"Number of images: {len(images)}")
                continue
        
        # Compute metrics
        metrics = self.compute_metrics(
            all_predictions, all_targets, all_question_types
        )
        results["metrics"] = metrics
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_suffix = f"_sample{self.sample_size}" if self.sample_size else ""
        output_file = os.path.join(
            self.output_dir,
            f"erqa_predictions{sample_suffix}_{timestamp}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {output_file}")
        
        return {
            "total_predictions": len(results["predictions"]),
            "output_file": output_file,
            **metrics
        }

    # --- Batch Processing Methods ---

    def supports_batch_processing(self) -> bool:
        """Enable batch processing for ERQA task."""
        return True

    def get_image_paths(self) -> List[str]:
        """Return a list of all unique local image paths required for the task."""
        if not self.data:
            self.load_data()
        
        all_image_paths = set()
        
        def add_path_recursive(path):
            if isinstance(path, str) and os.path.exists(path):
                all_image_paths.add(path)
            elif isinstance(path, (list, tuple)):
                for p in path:
                    add_path_recursive(p)
        
        for example in self.data:
            image_paths = example.get("image_paths", [])
            add_path_recursive(image_paths)
        
        return list(all_image_paths)

    def prepare_batch_data(self) -> List[Dict[str, Any]]:
        """Load and structure the ERQA dataset for batch processing."""
        if not self.data:
            self.load_data()
        
        batch_data = []
        for idx, example in enumerate(self.data):
            # Ensure image_paths is a list
            image_paths = example.get("image_paths", [])
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            
            # Apply adversarial mapping if available
            if hasattr(self, '_image_path_mapping'):
                effective_image_paths = [self._get_effective_image_path(path) for path in image_paths]
            else:
                effective_image_paths = image_paths
            
            batch_example = {
                "example_id": f"erqa_{idx}",
                "question_type": example.get("question_type"),
                "question": example.get("question"),
                "visual_indices": example.get("visual_indices", []),
                "image_paths": effective_image_paths,
                "ground_truth": example.get("answer"),
                "prompt": self.format_prompt(example)
            }
            batch_data.append(batch_example)
        
        return batch_data

    def format_batch_request(self, example: Dict[str, Any], gcs_image_prefix: str) -> Dict[str, Any]:
        """Format a single ERQA example for the Vertex AI batch prediction API."""
        # Use the existing format_prompt method to create the prompt
        prompt = example["prompt"]
        
        # Convert local image paths to GCS paths
        gcs_image_paths = []
        for local_path in example["image_paths"]:
            image_filename = os.path.basename(local_path)
            gcs_path = f"{gcs_image_prefix}/{image_filename}"
            gcs_image_paths.append(gcs_path)
        
        # Create the parts array - start with text prompt
        parts = [{"text": prompt}]
        
        # Add each image as a file_data part
        for gcs_image_path in gcs_image_paths:
            parts.append({
                "file_data": {
                    "file_uri": gcs_image_path,
                    "mime_type": "image/jpeg"
                }
            })
        
        return {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
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
        predictions = []
        with open(batch_results_path, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    # Extract the text response from the batch result
                    response_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    predictions.append(response_text.strip())
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Warning: Could not parse batch result line: {line}. Error: {e}")
                    predictions.append("")  # Add empty prediction for failed parsing
        
        # Verify we have matching numbers of predictions and metadata
        if len(predictions) != len(metadata):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(metadata)} metadata entries")
        
        # Extract ground truth and other info from metadata
        targets = [item["ground_truth"] for item in metadata]
        
        # If we have question type info in the original data, extract it for per-type metrics
        question_types = []
        if self.data:
            for example in self.data:
                question_types.append(example.get("question_type", "unknown"))
        else:
            question_types = ["unknown"] * len(targets)
        
        # Compute metrics using the existing compute_metrics method
        metrics = self.compute_metrics(predictions, targets, question_types[:len(predictions)])
        
        # Create detailed results
        results = {
            "metadata": {
                "model_name": "gemini-batch",
                "timestamp": datetime.now().isoformat(),
                "dataset": "ERQA",
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
        output_file = os.path.join(self.output_dir, f"erqa_batch_results_{timestamp}.json")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Batch evaluation results saved to {output_file}")
        print(f"Metrics: {metrics}")
        
        return metrics