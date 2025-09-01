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

    def evaluate_batch_results(self, batch_results_path: str, metadata_path: str = None) -> Dict[str, Any]:
        """Process completed batch job results and compute metrics."""
        if not self.data:
            self.load_data()
        
        def extract_image_key(filename):
            """Extract the key from both normal and adversarial image filenames."""
            # Remove extension first
            base_name = os.path.splitext(filename)[0]
            
            # Handle adversarial files: adv_image_69_0_erqa.jpg -> "69_0"
            if base_name.startswith("adv_image_"):
                # Remove "adv_image_" prefix and "_erqa" suffix
                key_part = base_name.replace("adv_image_", "")
                if key_part.endswith("_erqa"):
                    key_part = key_part[:-5]  # Remove "_erqa"
                return key_part
            
            # Handle normal files: image_0_0.jpg -> "0_0"
            elif base_name.startswith("image_"):
                return base_name.replace("image_", "")
            
            # Fallback: try to extract numbers from any filename
            else:
                import re
                # Look for pattern like "number_number" in the filename
                match = re.search(r'(\d+_\d+)', base_name)
                if match:
                    return match.group(1)
                
                # If no pattern found, try to extract just the first number
                match = re.search(r'(\d+)', base_name)
                if match:
                    return match.group(1)
            
            return None
        
        # Create a mapping from image file names to data examples
        image_to_example = {}
        for idx, example in enumerate(self.data):
            # Get the primary image path (first image)
            image_paths = example.get("image_paths", [])
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            
            if image_paths:
                # Extract the base filename and use it as key
                primary_image = image_paths[0]
                image_filename = os.path.basename(primary_image)
                key = extract_image_key(image_filename)
                
                if key:
                    image_to_example[key] = {
                        "index": idx,
                        "example": example,
                        "image_key": key,
                        "original_filename": image_filename
                    }
        
        print(f"Created mapping for {len(image_to_example)} examples")
        print(f"Sample mappings: {list(image_to_example.keys())[:5]}")
        
        # Load and parse batch results
        predictions_data = []
        with open(batch_results_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    result = json.loads(line.strip())
                    # Extract the text response from the batch result
                    response_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    
                    # Try to extract image key from the request if available
                    request = result.get('request', {})
                    image_key = None
                    extracted_filename = None
                    
                    # Look for image file references in the request
                    if 'contents' in request:
                        for content in request['contents']:
                            if 'parts' in content and content['parts']:
                                for part in content['parts']:
                                    if (part and 
                                        'file_data' in part and 
                                        part['file_data'] is not None and 
                                        isinstance(part['file_data'], dict) and
                                        'file_uri' in part['file_data']):
                                        file_uri = part['file_data']['file_uri']
                                        # Extract filename from file_uri
                                        extracted_filename = os.path.basename(file_uri)
                                        image_key = extract_image_key(extracted_filename)
                                        if image_key:
                                            break
                            if image_key:
                                break
                    
                    # If we couldn't extract image_key from request, use line number as fallback
                    if not image_key:
                        # Try to map by line number to data index
                        if line_num < len(self.data):
                            example = self.data[line_num]
                            image_paths = example.get("image_paths", [])
                            if image_paths:
                                primary_image = image_paths[0] if isinstance(image_paths, list) else image_paths
                                filename = os.path.basename(primary_image)
                                image_key = extract_image_key(filename)
                    
                    predictions_data.append({
                        "line_num": line_num,
                        "image_key": image_key,
                        "prediction": response_text.strip(),
                        "extracted_filename": extracted_filename
                    })
                    
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Warning: Could not parse batch result line {line_num}: {line}. Error: {e}")
                    predictions_data.append({
                        "line_num": line_num,
                        "image_key": None,
                        "prediction": "",
                        "extracted_filename": None
                    })
        
        print(f"Parsed {len(predictions_data)} batch results")
        print(f"Sample extracted keys: {[p['image_key'] for p in predictions_data[:5]]}")
        
        # Map predictions to examples and collect data for metrics
        all_predictions = []
        all_targets = []
        all_question_types = []
        results_details = []
        matched_count = 0
        unmatched_keys = []
        
        for pred_data in predictions_data:
            image_key = pred_data["image_key"]
            prediction = pred_data["prediction"]
            line_num = pred_data["line_num"]
            extracted_filename = pred_data["extracted_filename"]
            
            # Try to find matching example
            example_data = None
            match_method = "none"
            
            if image_key and image_key in image_to_example:
                example_data = image_to_example[image_key]
                match_method = "image_key"
                matched_count += 1
            elif line_num < len(self.data):
                # Fallback: use line number to match
                example_data = {
                    "index": line_num,
                    "example": self.data[line_num],
                    "image_key": f"line_{line_num}",
                    "original_filename": "fallback"
                }
                match_method = "line_number"
                matched_count += 1
            else:
                unmatched_keys.append({
                    "line_num": line_num,
                    "image_key": image_key,
                    "extracted_filename": extracted_filename
                })
            
            if example_data:
                example = example_data["example"]
                
                all_predictions.append(prediction)
                all_targets.append(example["answer"])
                all_question_types.append(example["question_type"])
                
                # Store detailed results (similar to evaluate method)
                result_entry = {
                    "line_num": line_num,
                    "image_key": example_data["image_key"],
                    "extracted_filename": extracted_filename,
                    "original_filename": example_data.get("original_filename"),
                    "question_type": example["question_type"],
                    "question": example["question"],
                    "ground_truth": example["answer"],
                    "model_prediction": prediction,
                    "visual_indices": example.get("visual_indices", []),
                    "matched_by": match_method
                }
                results_details.append(result_entry)
        
        print(f"Successfully matched {matched_count}/{len(predictions_data)} predictions to examples")
        if unmatched_keys:
            print(f"Unmatched keys sample: {unmatched_keys[:3]}")
        
        if not all_predictions:
            raise ValueError("No predictions could be matched to examples")
        
        # Compute metrics using the existing compute_metrics method
        metrics = self.compute_metrics(all_predictions, all_targets, all_question_types)
        
        # Create results structure similar to evaluate method
        results = {
            "metadata": {
                "model_name": "gemini-batch",
                "timestamp": datetime.now().isoformat(),
                "dataset": "ERQA",
                "metrics_config": self.metrics_config,
                "sample_size": self.sample_size,
                "random_seed": self.random_seed,
                "batch_processing": True,
                "total_examples": len(all_predictions),
                "matched_examples": matched_count,
                "total_batch_results": len(predictions_data),
                "unmatched_count": len(unmatched_keys)
            },
            "predictions": results_details,
            "metrics": metrics,
            "debug_info": {
                "sample_mapping_keys": list(image_to_example.keys())[:10],
                "sample_extracted_keys": [p['image_key'] for p in predictions_data[:10]],
                "unmatched_sample": unmatched_keys[:5] if unmatched_keys else []
            }
        }
        
        # Save detailed results
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_suffix = f"_sample{len(all_predictions)}" if len(all_predictions) != len(self.data) else ""
        output_file = os.path.join(
            self.output_dir,
            f"erqa_batch_predictions{sample_suffix}_{timestamp}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Batch evaluation results saved to {output_file}")
        print(f"Metrics: {metrics}")
        
        return {
            "total_predictions": len(results["predictions"]),
            "output_file": output_file,
            "matched_examples": matched_count,
            "unmatched_examples": len(unmatched_keys),
            **metrics
        }