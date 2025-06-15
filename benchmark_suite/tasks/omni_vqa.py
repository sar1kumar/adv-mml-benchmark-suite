import os
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
        
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format OmniMedVQA question with context and options if available"""
        prompt = "You are a medical expert. Please answer the following question about the medical image.\n\n"
        
        # Add dataset and modality information
        prompt += f"Dataset: {example['dataset']}\n"
        if "modality_type" in example:
            prompt += f"Modality: {example['modality_type']}\n"
        
        # Add question type and question
        prompt += f"Question Type: {example['question_type']}\n"
        prompt += f"Question: {example['question']}\n"
        
        # Add options if available
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            option_key = f'option_{opt}'
            if option_key in example:
                options.append(f"{opt}) {example[option_key]}")
        
        if options:
            prompt += "\nOptions:\n" + "\n".join(options) + "\n"
            prompt += "\nPlease select the correct option (A, B, C, or D)."
        else:
            prompt += "\nPlease provide a clear and concise answer based on the image."
        
        prompt += "\nAnswer: "
        return prompt
        
    def compute_metrics(self, predictions: List[str], targets: List[str],
                       question_types: List[str], datasets: List[str]) -> Dict[str, float]:
        """Compute metrics overall and per dataset/question type"""
        metrics = {}
        
        # Overall metrics
        for metric_name in self.metrics_config:
            if metric_name in self.metric_calculators:
                score = self.metric_calculators[metric_name].compute(predictions, targets)
                metrics[metric_name] = score
        
        # Per dataset metrics
        unique_datasets = set(datasets)
        for dataset in unique_datasets:
            dataset_indices = [i for i, d in enumerate(datasets) if d == dataset]
            dataset_preds = [predictions[i] for i in dataset_indices]
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
                "dataset": "OmniMedVQA",
                "access_type": self.access_type,
                "metrics_config": self.metrics_config,
                "sample_size": self.sample_size,
                "random_seed": self.random_seed
            },
            "predictions": []
        }
        
        all_predictions = []
        all_targets = []
        all_question_types = []
        all_datasets = []
        
        for example in self.data:
            # Construct image path
            image_path = os.path.join(self.data_path, example["image_path"])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                continue
            
            prompt = self.format_prompt(example)
            
            # Get model prediction
            prediction = model.process_image(image_path, prompt)
            model_answer = str(prediction.get("response", "")).strip()
            
            all_predictions.append(model_answer)
            all_targets.append(example["gt_answer"])
            all_question_types.append(example["question_type"])
            all_datasets.append(example["dataset"])
            
            # Store prediction details
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
        
        # Compute metrics
        metrics = self.compute_metrics(
            all_predictions, all_targets, all_question_types, all_datasets
        )
        results["metrics"] = metrics
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_suffix = f"_sample{self.sample_size}" if self.sample_size else ""
        output_file = os.path.join(
            self.output_dir,
            f"omnimed_vqa_{self.access_type}{sample_suffix}_{timestamp}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {output_file}")
        
        return {
            "total_predictions": len(results["predictions"]),
            "output_file": output_file,
            **metrics
        }
