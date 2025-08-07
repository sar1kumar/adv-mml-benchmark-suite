import os
import json
import random
from typing import Dict, List, Any, Tuple
from datetime import datetime
from benchmark_suite.tasks.base_task import BaseTask
from benchmark_suite.models.base_model import BaseModel

class VQARADTask(BaseTask):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = config["data_path"]
        self.split = config.get("split", ["train", "val", "test"])
        self.output_dir = config.get("output_dir", "results")
        self.sample_size = config.get("sample_size", None)
        self.random_seed = config.get("random_seed", 42)
        self.data = {}
        self.question_types = ["PRES", "MODALITY", "ORGAN", "POS", "ABN"]
        self.answer_types = ["CLOSED", "OPEN"]
        
    def load_data(self) -> None:
        """Load VQA-RAD dataset with optional sampling"""
        json_file = os.path.join(self.data_path, "VQA_RAD Dataset Public.json")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"VQA-RAD data not found at {json_file}")
            
        with open(json_file, 'r') as f:
            all_data = json.load(f)
            
        # Split data based on phrase_type
        for item in all_data:
            split_name = item["phrase_type"].replace("_freeform", "")
            if split_name not in self.data:
                self.data[split_name] = []
            self.data[split_name].append(item)
        
        # Add Sample dataset
        if self.sample_size is not None:
            random.seed(self.random_seed)
            sampled_data = {}
            
            for split_name, split_data in self.data.items():
                if split_name not in self.split:
                    continue
                    
                if self.sample_size > len(split_data):
                    print(f"Warning: Requested sample size {self.sample_size} is larger than dataset size {len(split_data)} for split {split_name}. Using full split.")
                    sampled_data[split_name] = split_data
                else:
                    sampled_indices = random.sample(range(len(split_data)), self.sample_size)
                    sampled_data[split_name] = [split_data[i] for i in sampled_indices]
                    print(f"Sampled {self.sample_size} examples from {split_name} split")
            
            self.data = sampled_data
                
    def format_prompt(self, example: Dict[str, Any]) -> str:    #TODO - add other prompt formats for different phrase_types and modalities
        """Format VQA-RAD question with context about the image"""
        prompt = "You are a medical imaging expert. Answer the following question about the radiology image.\n\n"
        prompt += f"Image Organ: {example['image_organ']}\n"
        prompt += f"Question Type: {example['question_type']}\n"
        prompt += f"Question: {example['question']}\n"
        
        if example['answer_type'] == 'CLOSED':
            prompt += "This is a yes/no question. Answer with 'Yes' or 'No' only.\n"
        else:
            prompt += "Provide a concise and accurate answer.\n"
            
        prompt += "Answer: "
        return prompt
        
        
    def compute_metrics(self, predictions_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute metrics with breakdown by question and answer types"""
        if not predictions_data:
            metrics = {'accuracy': 0.0}
            for q_type in self.question_types:
                metrics[f'{q_type.lower()}_accuracy'] = 0.0
            for a_type in self.answer_types:
                metrics[f'{a_type.lower()}_accuracy'] = 0.0
            return metrics

        metrics = {}

        predictions = [str(item.get("model_prediction", {}).get("response", "")).strip() for item in predictions_data]
        targets = [str(item["ground_truth"]).strip() for item in predictions_data]
        question_types = [item["question_type"] for item in predictions_data]
        answer_types = [item["answer_type"] for item in predictions_data]
        
        # Overall accuracy
        correct = sum(p == t for p, t in zip(predictions, targets))
        metrics['accuracy'] = correct / len(predictions) if predictions else 0.0
        
        # Per question type accuracy
        for q_type in self.question_types:
            q_type_indices = [i for i, qt in enumerate(question_types) if qt == q_type]
            if q_type_indices:
                q_type_preds = [predictions[i] for i in q_type_indices]
                q_type_targets = [targets[i] for i in q_type_indices]
                correct = sum(p == t for p, t in zip(q_type_preds, q_type_targets))
                metrics[f'{q_type.lower()}_accuracy'] = correct / len(q_type_indices)
            else:
                metrics[f'{q_type.lower()}_accuracy'] = 0.0
        
        # Per answer type accuracy
        for a_type in self.answer_types:
            a_type_indices = [i for i, at in enumerate(answer_types) if at == a_type]
            if a_type_indices:
                a_type_preds = [predictions[i] for i in a_type_indices]
                a_type_targets = [targets[i] for i in a_type_indices]
                correct = sum(p == t for p, t in zip(a_type_preds, a_type_targets))
                metrics[f'{a_type.lower()}_accuracy'] = correct / len(a_type_indices)
            else:
                metrics[f'{a_type.lower()}_accuracy'] = 0.0
        
        return metrics
        
    def evaluate(self, model: BaseModel) -> Dict[str, Any]:
        """Generate predictions for VQA-RAD task and store in JSON file"""
        if not self.data:
            self.load_data()
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Prepare results structure
        results = {
            "metadata": {
                "model_name": model.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "dataset": "VQA-RAD",
                "splits": self.split
            },
            "predictions": []
        }
        
        for split_name in self.split:
            if split_name not in self.data:
                continue
                
            split_data = self.data[split_name]
            
            for example in split_data:
                # Get image path from the image name
                image_path = os.path.join(self.data_path, "VQA_RAD Image Folder", example["image_name"])
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}")
                    continue
                    
                prompt = self.format_prompt(example)
                
                # Get model prediction
                prediction = model.process_image(image_path, prompt)
                
                model_answer = str(prediction.get("response", "")).strip()
                # Store prediction and ground truth
                result_entry = {
                    "split": split_name,
                    "qid": example["qid"],
                    "image_name": example["image_name"],
                    "image_organ": example["image_organ"],
                    "question_type": example["question_type"],
                    "answer_type": example["answer_type"],
                    "question": example["question"],
                    "ground_truth": example["answer"],
                    "model_prediction": prediction,
                    "prompt": prompt,
                    "time_metrics": prediction.get("ollama_metrics", {})
                }
                results["predictions"].append(result_entry)
        
        # Compute metrics
        metrics = self.compute_metrics(results["predictions"])
        results["metrics"] = metrics
        # Save results to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"vqa_rad_predictions_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Predictions saved to {output_file}")
        
        # Return basic stats
        return {
            "total_predictions": len(results["predictions"]),
            "output_file": output_file,
            #TODO: Add evalution metrics here (accuracy, Bleu score, etc.) computed from the predictions and ground truths.
            "metrics": metrics
        }