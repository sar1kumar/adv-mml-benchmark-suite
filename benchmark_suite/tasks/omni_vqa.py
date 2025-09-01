import os
import re
import json
import random
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
from benchmark_suite.tasks.base_task import BaseTask
from benchmark_suite.models.base_model import BaseModel
from benchmark_suite.metrics import BLEU, METEOR, ROUGEL, CIDEr, Accuracy

class OmniMedVQATask(BaseTask):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = config["data_path"]
        self.split = config.get("split", ["test"])
        self.output_dir = config.get("output_dir", "results")
        self.sample_size = config.get("sample_size", None)
        self.random_seed = config.get("random_seed", 42)
        self.metrics_config = config.get("metrics", ["accuracy", "bleu", "rouge"])
        self.access_type = config.get("access_type", "Open-access")  # or "Restricted-access"
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
        """Load OmniMedVQA dataset with optional sampling"""
        qa_dir = os.path.join(self.data_path, "QA_information", self.access_type)
        if not os.path.exists(qa_dir):
            raise FileNotFoundError(f"QA information directory not found at {qa_dir}")
        
        all_data = []
        
        # Load all JSON files from the specified access type directory
        json_files = list(Path(qa_dir).glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {qa_dir}")
        
        print(f"Found {len(json_files)} JSON files in {qa_dir}")
        
        for json_file in json_files:
            print(f"Processing {json_file.name}...")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    dataset_data = json.load(f)
                    # Check if dataset_data is a list or needs to be wrapped
                    if isinstance(dataset_data, dict):
                        # If it's a dictionary, assume it contains a list of questions
                        if "questions" in dataset_data:
                            dataset_data = dataset_data["questions"]
                        else:
                            # Convert dict items to list if it's a dict of questions
                            dataset_data = list(dataset_data.values())
                    
                    # Add dataset name to each item if not present
                    dataset_name = json_file.stem  # Get filename without extension
                    for item in dataset_data:
                        if "dataset" not in item:
                            item["dataset"] = dataset_name
                    
                    print(f"Loaded {len(dataset_data)} questions from {json_file.name}")
                    all_data.extend(dataset_data)
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file.name}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing {json_file.name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data was loaded from any JSON files")
        
        print(f"Total questions loaded: {len(all_data)}")
        
        # Group data by dataset
        dataset_groups = {}
        for item in all_data:
            dataset = item["dataset"]
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(item)
        
        print("\nDataset statistics:")
        for dataset, items in dataset_groups.items():
            print(f"{dataset}: {len(items)} questions")
        
        # Apply sampling if specified
        if self.sample_size is not None:
            random.seed(self.random_seed)
            sampled_data = []
            
            for dataset, items in dataset_groups.items():
                if self.sample_size > len(items):
                    print(f"Warning: Requested sample size {self.sample_size} is larger than dataset size {len(items)} for {dataset}. Using all available examples.")
                    sampled_data.extend(items)
                else:
                    sampled_items = random.sample(items, self.sample_size)
                    sampled_data.extend(sampled_items)
                    print(f"Sampled {self.sample_size} examples from {dataset}")
            
            self.data = sampled_data
            print(f"\nTotal sampled questions: {len(self.data)}")
        else:
            self.data = all_data
            print(f"\nUsing all {len(self.data)} questions")

    def find_image_with_alternate_extensions(self, base_image_path: str) -> str:
        """
        Try to find the image with different extensions.
        
        Args:
            base_image_path: Original image path
            
        Returns:
            str: Path to found image or None if not found
        """
        # Remove any existing extension
        base_path = os.path.splitext(base_image_path)[0]
        
        # List of extensions to try (in order of preference)
        extensions = ['.jpeg', '.jpg']
        
        # Try each extension
        for ext in extensions:
            test_path = base_path + ext
            if os.path.exists(test_path):
                return test_path
        
        return None
    
    def extract_options_from_prompt(self, prompt: str) -> Dict[str, str]:
        """Extract options mapping from the prompt.
        
        Args:
            prompt: The full prompt containing options like "A) word B) word2..."
            
        Returns:
            Dictionary mapping option letters to their values
        """
        # Look for patterns like "A) word" or "A: word" or "A. word"
        option_pattern = r'([A-D])[):\.\s]\s*([^\n]+)'
        matches = re.finditer(option_pattern, prompt, re.IGNORECASE)
        
        options = {}
        for match in matches:
            letter = match.group(1).upper()  # Normalize to uppercase
            value = match.group(2).strip()
            options[letter] = value
            
        return options

    def extract_model_answer(self, prediction: str) -> str:
        """Extract the letter answer (A,B,C,D) from model prediction.
        
        Args:
            prediction: Raw model prediction string
            
        Returns:
            Extracted letter answer or None if not found
        """
        # Common patterns for letter answers
        patterns = [
            r'^([A-Da-d])[):\.\s]',  # A) or A: or A.
            r'^option\s+([A-Da-d])',  # Option A
            r'^\(?([A-Da-d])\)?$',   # Just A or (A)
            r'([A-Da-d])[):\.\s].*$'  # Matches letter anywhere with separator
        ]
        
        prediction = prediction.strip()
        for pattern in patterns:
            match = re.search(pattern, prediction, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None

    def process_prediction(self, prediction: str, options: Dict[str, str]) -> str:
        """Process a prediction based on whether it's a multiple choice or free-form question.
        
        Args:
            prediction: Raw model prediction
            options: Dictionary of options (may be empty for free-form questions)
            
        Returns:
            Processed prediction
        """
        # If no options or empty options, it's a free-form question
        if not options or all(not value for value in options.values()):
            return prediction.strip()
        
        # For multiple choice, try to extract and map the letter answer
        letter_answer = self.extract_model_answer(prediction)
        if letter_answer and letter_answer in options:
            return options[letter_answer]
        
        # If we couldn't extract a letter but it's multiple choice,
        # try to match the prediction text against option values
        pred_lower = prediction.lower().strip()
        for option_text in options.values():
            if option_text.lower() in pred_lower:
                return option_text
            
        # If all else fails, return the original prediction
        return prediction.strip()

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format OmniMedVQA question with context and options if available"""
        prompt = "You are a medical expert. Please answer the following question about the medical image.\n\n"
        
        # Add question type and question
        prompt += f"Question: {example['question']}\n"
        
        # Add options if available
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            option_key = f'option_{opt}'
            if option_key in example:
                options.append(f"{opt}) {example[option_key]}")
        
        if options:
            prompt += "\nOptions:\n" + "\n".join(options) + "\n"
            prompt += "\nPlease select the correct option (A, B, C, or D). Dont add any other text or explanation. Please answer directly with only the letter of the correct option and nothing else."
        else:
            prompt += "\nPlease provide a clear and concise answer in few words based on the image."
        
        prompt += "\nAnswer: "
        return prompt
        
    def compute_metrics(self, predictions: List[str], targets: List[str],
                   question_types: List[str], datasets: List[str], 
                   options_list: List[Dict[str, str]]) -> Dict[str, float]:
        """Compute metrics overall and per dataset/question type
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth answers
            question_types: List of question types
            datasets: List of dataset names
            options_list: List of option dictionaries (may be empty for free-form questions)
        """
        metrics = {}
        
        # Process predictions based on whether they're multiple choice or free-form
        processed_predictions = [
            self.process_prediction(pred, opts) 
            for pred, opts in zip(predictions, options_list)
        ]
        
        # Overall metrics
        for metric_name in self.metrics_config:
            if metric_name in self.metric_calculators:
                score = self.metric_calculators[metric_name].compute(
                    processed_predictions, targets
                )
                metrics[metric_name] = score
        
        # Per dataset metrics
        unique_datasets = set(datasets)
        for dataset in unique_datasets:
            dataset_indices = [i for i, d in enumerate(datasets) if d == dataset]
            dataset_preds = [processed_predictions[i] for i in dataset_indices]
            dataset_targets = [targets[i] for i in dataset_indices]
            
            for metric_name in self.metrics_config:
                if metric_name in self.metric_calculators:
                    score = self.metric_calculators[metric_name].compute(
                        dataset_preds, dataset_targets
                    )
                    metrics[f"{dataset}_{metric_name}"] = score
        
        # Per question type metrics
        unique_types = set(question_types)
        for q_type in unique_types:
            type_indices = [i for i, qt in enumerate(question_types) if qt == q_type]
            type_preds = [processed_predictions[i] for i in type_indices]
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
                "dataset": "OmniMedVQA",
                "access_type": self.access_type,
                "metrics_config": self.metrics_config,
                "sample_size": self.sample_size,
                "random_seed": self.random_seed
            },
            "predictions": [],
            "errors": []  # Track errors for reporting
        }
        
        all_predictions = []
        all_targets = []
        all_question_types = []
        all_datasets = []
        all_options = []
        
        total_questions = len(self.data)
        processed = 0
        errors = 0
        
        for example in self.data:
            processed += 1
            if processed % 10 == 0:  # Progress update every 10 questions
                print(f"Processing {processed}/{total_questions} questions... (Errors: {errors})")
                
            try:
                # Construct image path
                image_path = os.path.join(self.data_path, example["image_path"])
                if not os.path.exists(image_path):
                    image_path = self.find_image_with_alternate_extensions(image_path)
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image not found at {image_path}")
                
                prompt = self.format_prompt(example)
                
                try:
                    # Get model prediction with timeout
                    prediction = model.process_image(image_path, prompt)
                    model_answer = str(prediction.get("response", "")).strip()
                    
                    if not model_answer:  # Handle empty responses
                        raise ValueError("Empty response from model")
                    
                except Exception as api_error:
                    # Log API error and continue with next question
                    error_entry = {
                        "question_id": example["question_id"],
                        "error_type": "API_ERROR",
                        "error_message": str(api_error),
                        "dataset": example["dataset"]
                    }
                    results["errors"].append(error_entry)
                    errors += 1
                    print(f"API Error for question {example['question_id']}: {str(api_error)}")
                    continue
                
                # Store successful prediction
                result_entry = {
                    "question_id": example["question_id"],
                    "dataset": example["dataset"],
                    "question_type": example["question_type"],
                    "question": example["question"],
                    "options": {
                        "A": example.get("option_A", ""),
                        "B": example.get("option_B", ""),
                        "C": example.get("option_C", ""),
                        "D": example.get("option_D", "")
                    },
                    "ground_truth": example["gt_answer"],
                    "model_prediction": model_answer,
                    "prompt": prompt,
                    "modality_type": example.get("modality_type", ""),
                    "time_metrics": prediction.get("ollama_metrics", {})
                }
                
                results["predictions"].append(result_entry)
                
                # Collect data for metrics computation
                all_predictions.append(model_answer)
                all_targets.append(example["gt_answer"])
                all_question_types.append(example["question_type"])
                all_datasets.append(example["dataset"])
                all_options.append(result_entry["options"])
                
            except Exception as e:
                # Log any other errors but continue processing
                error_entry = {
                    "question_id": example.get("question_id", "unknown"),
                    "error_type": "PROCESSING_ERROR",
                    "error_message": str(e),
                    "dataset": example.get("dataset", "unknown")
                }
                results["errors"].append(error_entry)
                errors += 1
                print(f"Error processing question: {str(e)}")
                continue
        
        # Compute metrics only if we have predictions
        if all_predictions:
            try:
                metrics = self.compute_metrics(
                    all_predictions, all_targets, all_question_types, all_datasets, all_options
                )
                results["metrics"] = metrics
            except Exception as e:
                print(f"Error computing metrics: {str(e)}")
                results["metrics"] = {"error": str(e)}
        else:
            results["metrics"] = {"error": "No successful predictions to compute metrics"}
        
        # Add error summary to metadata
        results["metadata"]["total_questions"] = total_questions
        results["metadata"]["successful_predictions"] = len(all_predictions)
        results["metadata"]["total_errors"] = errors
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_suffix = f"_sample{self.sample_size}" if self.sample_size else ""
        output_file = os.path.join(
            self.output_dir,
            f"omnimed_vqa_{self.access_type}{sample_suffix}_{timestamp}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nEvaluation Complete:")
        print(f"Total questions: {total_questions}")
        print(f"Successful predictions: {len(all_predictions)}")
        print(f"Total errors: {errors}")
        print(f"Results saved to {output_file}")
        
        return {
            "total_questions": total_questions,
            "successful_predictions": len(all_predictions),
            "total_errors": errors,
            "output_file": output_file,
            **(results.get("metrics", {}))
        }

    def supports_batch_processing(self) -> bool:
        """Enable batch processing for OmniMed VQA task."""
        return True

    def get_image_paths(self) -> List[str]:
        """Return a list of all unique local image paths required for the task."""
        if not self.data:
            self.load_data()
        
        all_image_paths = set()
        images_dir = os.path.join(self.data_path, "images")
        
        for example in self.data:
            # Convert image filename to full path
            image_filename = example.get("image_path", "")
            if image_filename:
                # Handle different image extensions (convert TIF/BMP to JPG as in omni_batch_inputs.py)
                image_filename = self._convert_to_jpg_path(image_filename)
                full_path = os.path.join(images_dir, image_filename)
                if os.path.exists(full_path):
                    all_image_paths.add(full_path)
        
        return list(all_image_paths)

    def _convert_to_jpg_path(self, image_path: str) -> str:
        """Convert TIF/BMP file extensions to JPG (from omni_batch_inputs.py)"""
        lower_path = image_path.lower()
        if lower_path.endswith(('.tif', '.tiff', '.bmp')):
            base_path = image_path.rsplit('.', 1)[0]
            return f"{base_path}.jpg"
        return image_path

    def prepare_batch_data(self) -> List[Dict[str, Any]]:
        """Load and structure the OmniMed dataset for batch processing."""
        if not self.data:
            self.load_data()
        
        batch_data = []
        for idx, example in enumerate(self.data):
            # Convert image path
            image_filename = self._convert_to_jpg_path(example.get("image_path", ""))
            
            batch_example = {
                "example_id": example.get("question_id"),  # Add explicit question_id field
                "dataset": example.get("dataset"),
                "question_type": example.get("question_type"),
                "question": example.get("question"),
                "image_path": image_filename,
                "ground_truth": example.get("gt_answer"),
                "options": {
                    "A": example.get("option_A", ""),
                    "B": example.get("option_B", ""),
                    "C": example.get("option_C", ""),
                    "D": example.get("option_D", "")
                },
                "modality_type": example.get("modality_type", ""),
                "prompt": self.format_prompt(example)
            }
            batch_data.append(batch_example)
        
        return batch_data

    def format_batch_request(self, example: Dict[str, Any], gcs_image_prefix: str) -> Dict[str, Any]:
        """Format a single OmniMed example for the Vertex AI batch prediction API."""
        prompt = example["prompt"]
        gcs_image_path = f"{gcs_image_prefix}/{example['image_path']}"
        
        return {
            "custom_id": example["example_id"],  # Add custom_id for proper tracking
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
        
        # Create mapping from question_id to metadata
        metadata_map = {item["question_id"]: item for item in metadata if "question_id" in item}
        
        # Load and parse batch results
        predictions = []
        targets = []
        question_types = []
        datasets = []
        options_list = []
        matched_count = 0
        
        with open(batch_results_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    result = json.loads(line.strip())
                    
                    # Extract response
                    response_text = result.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    
                    # Try to get question_id from custom_id if available
                    question_id = result.get('custom_id')
                    
                    if question_id and question_id in metadata_map:
                        # Use metadata mapping for accurate ground truth
                        meta_item = metadata_map[question_id]
                        predictions.append(response_text.strip())
                        targets.append(meta_item["ground_truth"])
                        
                        # Get additional info from original data if available
                        original_data = meta_item.get("original_data", {})
                        question_types.append(original_data.get("question_type", "unknown"))
                        datasets.append(original_data.get("dataset", "unknown"))
                        options_list.append(original_data.get("options", {}))
                        matched_count += 1
                        
                    elif line_num < len(metadata):
                        # Fallback to line order (less reliable)
                        meta_item = metadata[line_num]
                        predictions.append(response_text.strip())
                        targets.append(meta_item["ground_truth"])
                        
                        original_data = meta_item.get("original_data", {})
                        question_types.append(original_data.get("question_type", "unknown"))
                        datasets.append(original_data.get("dataset", "unknown"))
                        options_list.append(original_data.get("options", {}))
                    else:
                        print(f"Warning: Could not match result at line {line_num}")
                        continue
                        
                except Exception as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue
        
        print(f"Matched {matched_count}/{len(predictions)} results using question_id")
        
        # Verify we have data to evaluate
        if not predictions:
            raise ValueError("No valid predictions found for evaluation")
        
        # Compute metrics using the existing compute_metrics method
        metrics = self.compute_metrics(predictions, targets, question_types, datasets, options_list)
        
        # Create detailed results
        results = {
            "metadata": {
                "model_name": "gemini-batch",
                "timestamp": datetime.now().isoformat(),
                "dataset": "OmniMedVQA",
                "access_type": self.access_type,
                "total_examples": len(predictions),
                "matched_by_id": matched_count,
                "batch_processing": True
            },
            "metrics": metrics,
            "summary": {
                "total_predictions": len(predictions),
                "matched_by_question_id": matched_count,
                "processing_mode": "batch"
            }
        }
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"omnimed_batch_results_{timestamp}.json")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Batch evaluation results saved to {output_file}")
        print(f"Metrics: {metrics}")
        
        return metrics
