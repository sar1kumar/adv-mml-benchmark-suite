import os
import json
import pandas as pd
from typing import Dict, List, Any
from .base_task import BaseTask
from ..models.base_model import BaseModel

class MMLUTask(BaseTask):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = config["data_path"]
        self.subjects = config.get("subjects", ["mathematics", "computer_science", "physics", "medicine"])
        self.few_shot_k = config.get("few_shot_k", 5)
        self.data = {}
        
    def load_data(self) -> None:
        """Load MMLU dataset for specified subjects"""
        for subject in self.subjects:
            subject_path = os.path.join(self.data_path, f"{subject}.csv")
            if not os.path.exists(subject_path):
                raise FileNotFoundError(f"MMLU data not found at {subject_path}")
                
            df = pd.read_csv(subject_path)
            self.data[subject] = {
                "questions": df["question"].tolist(),
                "choices": df[["A", "B", "C", "D"]].values.tolist(),
                "answers": df["answer"].tolist()
            }
            
    def format_prompt(self, example: Dict[str, Any], few_shot_examples: List[Dict] = None) -> str:
        """Format MMLU question with few-shot examples if provided"""
        prompt = "Answer the following multiple choice question. Choose the most appropriate answer from A, B, C, or D.\n\n"
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for ex in few_shot_examples:
                prompt += f"Question: {ex['question']}\n"
                for i, choice in enumerate(ex['choices']):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += f"Answer: {ex['answer']}\n\n"
        
        # Add the actual question
        prompt += f"Question: {example['question']}\n"
        for i, choice in enumerate(example['choices']):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer: "
        
        return prompt
        
    def evaluate(self, model: BaseModel) -> Dict[str, float]:
        """Evaluate model on MMLU task"""
        if not self.data:
            self.load_data()
            
        all_predictions = []
        all_targets = []
        
        for subject, subject_data in self.data.items():
            for i in range(len(subject_data["questions"])):
                # Get few-shot examples (excluding current question)
                few_shot_indices = list(range(len(subject_data["questions"])))
                few_shot_indices.remove(i)
                few_shot_examples = [
                    {
                        "question": subject_data["questions"][j],
                        "choices": subject_data["choices"][j],
                        "answer": subject_data["answers"][j]
                    }
                    for j in few_shot_indices[:self.few_shot_k]
                ]
                
                # Prepare current example
                example = {
                    "question": subject_data["questions"][i],
                    "choices": subject_data["choices"][i],
                }
                
                # Get model prediction
                prompt = self.format_prompt(example, few_shot_examples)
                prediction = model.generate(prompt).strip()
                
                # Extract single-letter answer if model gave explanation
                if len(prediction) > 1:
                    for letter in ["A", "B", "C", "D"]:
                        if letter in prediction:
                            prediction = letter
                            break
                
                all_predictions.append(prediction)
                all_targets.append(subject_data["answers"][i])
        
        # Compute metrics
        results = self.compute_metrics(all_predictions, all_targets)
        
        # Add per-subject metrics
        subject_results = {}
        start_idx = 0
        for subject in self.subjects:
            subject_len = len(self.data[subject]["questions"])
            subject_preds = all_predictions[start_idx:start_idx + subject_len]
            subject_targets = all_targets[start_idx:start_idx + subject_len]
            subject_results[f"{subject}_accuracy"] = sum(p == t for p, t in zip(subject_preds, subject_targets)) / subject_len
            start_idx += subject_len
            
        results.update(subject_results)
        return results 