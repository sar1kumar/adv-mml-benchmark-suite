import os
import json
import numpy as np
from typing import Dict, List, Any
from .base_task import BaseTask
from ..models.base_model import BaseModel

class EmbodiedVQATask(BaseTask):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = config["data_path"]
        self.split = config.get("split", ["train", "val"])
        self.data = {}
        
    def load_data(self) -> None:
        """Load EmbodiedVQA dataset"""
        for split_name in self.split:
            split_file = os.path.join(self.data_path, f"{split_name}.json")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"EmbodiedVQA data not found at {split_file}")
                
            with open(split_file, 'r') as f:
                self.data[split_name] = json.load(f)
                
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format EmbodiedVQA question with context"""
        prompt = "Answer the following question about the scene, taking into account the visual information and context provided.\n\n"
        
        if "scene_description" in example:
            prompt += f"Scene Description: {example['scene_description']}\n\n"
            
        if "context" in example:
            prompt += f"Context: {example['context']}\n\n"
            
        prompt += f"Question: {example['question']}\n"
        
        if "options" in example:
            prompt += "Options:\n"
            for i, option in enumerate(example["options"]):
                prompt += f"{chr(65+i)}. {option}\n"
                
        prompt += "Answer: "
        return prompt
        
    def evaluate(self, model: BaseModel) -> Dict[str, float]:
        """Evaluate model on EmbodiedVQA task"""
        if not self.data:
            self.load_data()
            
        all_predictions = []
        all_targets = []
        
        for split_name in self.split:
            split_data = self.data[split_name]
            
            for example in split_data:
                # Get paths to all relevant images for this example
                image_paths = []
                for view_id in example["view_ids"]:
                    image_path = os.path.join(self.data_path, "images", f"{view_id}.jpg")
                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                
                if not image_paths:
                    continue  # Skip if no images found
                    
                prompt = self.format_prompt(example)
                
                # Process each image and aggregate predictions
                predictions = []
                for image_path in image_paths:
                    pred = model.process_image(image_path, prompt).strip()
                    predictions.append(pred)
                
                # Take most common prediction or first one if all different
                from collections import Counter
                prediction = Counter(predictions).most_common(1)[0][0]
                
                # For multiple choice, extract letter if needed
                if "options" in example:
                    if len(prediction) > 1:
                        for letter in ["A", "B", "C", "D"]:
                            if letter in prediction:
                                prediction = letter
                                break
                
                all_predictions.append(prediction)
                all_targets.append(example["answer"])
        
        # Compute metrics
        results = self.compute_metrics(all_predictions, all_targets)
        
        # Add per-split metrics
        split_results = {}
        start_idx = 0
        for split_name in self.split:
            split_len = len([ex for ex in self.data[split_name] if any(os.path.exists(os.path.join(self.data_path, "images", f"{v}.jpg")) for v in ex["view_ids"])])
            split_preds = all_predictions[start_idx:start_idx + split_len]
            split_targets = all_targets[start_idx:start_idx + split_len]
            
            for metric in self.metrics:
                if metric == "accuracy":
                    split_results[f"{split_name}_accuracy"] = sum(p.strip().lower() == t.strip().lower() 
                                                                for p, t in zip(split_preds, split_targets)) / split_len
                elif metric == "rouge":
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                    scores = []
                    for pred, target in zip(split_preds, split_targets):
                        score = scorer.score(target.lower(), pred.lower())
                        scores.append(score['rougeL'].fmeasure)  # Using ROUGE-L F1
                    split_results[f"{split_name}_rouge"] = sum(scores) / len(scores)
                    
            start_idx += split_len
            
        results.update(split_results)
        return results 