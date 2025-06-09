import os
import json
import random
from typing import Dict, List, Any, Tuple
from datetime import datetime
from benchmark_suite.tasks.base_task import BaseTask
from benchmark_suite.models.base_model import BaseModel
from benchmark_suite.utils.prompts.sme_prompts import create_prompt

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
        
    def compute_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Compute metrics for SME task"""
        metrics = {}
        
        # Overall accuracy
        correct = sum(p.strip().lower() == t.strip().lower() for p, t in zip(predictions, targets))
        metrics['accuracy'] = correct / len(predictions)
        
        return metrics
        
    def evaluate(self, model: BaseModel) -> Dict[str, float]:
        """Generate predictions for SME task and store in JSON file"""
        if not self.data:
            self.load_data()
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Prepare results structure
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
        
        all_predictions = []
        all_targets = []
        
        for split_name in self.split:
            if split_name not in self.data:
                continue
                
            split_data = self.data[split_name]
            print(f"Processing {len(split_data)} examples from {split_name} split")
            
            # Iterate through the dictionary items
            for example_id, example_data in split_data.items():
                # Construct image filename by adding .jpg to imageId
                image_filename = f"{example_data['imageId']}.jpg"
                image_path = os.path.join(self.data_path, image_filename)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}")
                    continue
                    
                prompt = self.format_prompt(example_data)
                
                # Get model prediction
                prediction = model.process_image(image_path, prompt)
                print(prediction)
                model_answer = str(prediction.get("response", "")).strip()
                all_predictions.append(model_answer)
                all_targets.append(example_data["answer"])
                
                # Store prediction and ground truth
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
        #metrics = self.compute_metrics(all_predictions, all_targets)
        #results["metrics"] = metrics
        
        # Save results to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_suffix = f"_sample{self.sample_size}" if self.sample_size else ""
        output_file = os.path.join(self.output_dir, f"sme_predictions{sample_suffix}_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Predictions saved to {output_file}")
        
        # Return metrics and stats
        return {
            "total_predictions": len(results["predictions"]),
            "output_file": output_file,
            #**metrics
        }