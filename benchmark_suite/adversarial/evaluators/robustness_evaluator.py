import time
import json
from typing import Dict, List, Any, Tuple

import numpy as np

from benchmark_suite.adversarial.generators.base_generator import BaseAdversarialGenerator
from benchmark_suite.models.base_model import BaseModel
from benchmark_suite.tasks.base_task import BaseTask

class RobustnessEvaluator:
    """Evaluate model robustness against adversarial examples"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_trials = config.get("num_trials", 5)
        self.metrics = config.get("metrics", ["accuracy", "robustness", "inference_time"])
        
    def evaluate(self, model: BaseModel, task: BaseTask, 
                generator: BaseAdversarialGenerator) -> Dict[str, Any]:
        """Run adversarial evaluation"""
        results = {
            "clean_performance": {},
            "adversarial_performance": {},
            "robustness_metrics": {},
            "timing_metrics": {}
        }
        
        # Evaluate on clean data
        clean_metrics, clean_timing = self._evaluate_performance(model, task)
        results["clean_performance"] = clean_metrics
        results["timing_metrics"]["clean"] = clean_timing
        
        # Generate and evaluate on adversarial data
        adv_metrics = []
        adv_timing = []
        
        for trial in range(self.num_trials):
            # Generate adversarial examples
            task_data = self._get_task_data(task)
            perturbed_data = self._generate_adversarial_data(generator, task_data)
            
            # Evaluate on adversarial examples
            trial_metrics, trial_timing = self._evaluate_performance(
                model, task, perturbed_data
            )
            
            adv_metrics.append(trial_metrics)
            adv_timing.append(trial_timing)
            
        # Aggregate adversarial results
        results["adversarial_performance"] = self._aggregate_metrics(adv_metrics)
        results["timing_metrics"]["adversarial"] = self._aggregate_timing(adv_timing)
        
        # Compute robustness metrics
        results["robustness_metrics"] = self._compute_robustness_metrics(
            results["clean_performance"],
            results["adversarial_performance"]
        )
        
        return results
        
    def _evaluate_performance(self, model: BaseModel, task: BaseTask, 
                            data: Dict[str, Any] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate model performance and timing"""
        performance_metrics = {}
        timing_metrics = {
            "total_time": 0,
            "mean_inference_time": 0,
            "std_inference_time": 0
        }
        
        start_time = time.time()
        inference_times = []
        
        if data is None:
            # Use task's original evaluate method
            results = task.evaluate(model)
            performance_metrics = {k: v for k, v in results.items() 
                                if k != "elapsed_time"}
            timing_metrics["total_time"] = results.get("elapsed_time", 0)
        else:
            # Evaluate on provided data
            predictions = []
            targets = []
            
            for example in data["examples"]:
                pred_start = time.time()
                if "image_path" in example:
                    pred = model.process_image(
                        example["image_path"], 
                        example["question"]
                    )
                else:
                    pred = model.generate(example["text"])
                pred_time = time.time() - pred_start
                
                predictions.append(pred)
                targets.append(example["target"])
                inference_times.append(pred_time)
                
            performance_metrics = task.compute_metrics(predictions, targets)
            timing_metrics["total_time"] = time.time() - start_time
            
        if inference_times:
            timing_metrics["mean_inference_time"] = np.mean(inference_times)
            timing_metrics["std_inference_time"] = np.std(inference_times)
            
        return performance_metrics, timing_metrics
        
    def _get_task_data(self, task: BaseTask) -> Dict[str, Any]:
        """Extract evaluation data from task"""
        # This is a simplified version - actual implementation would need
        # to handle different task types and data formats
        data = {"examples": []}
        
        if hasattr(task, "data"):
            if isinstance(task.data, dict):
                # Handle MMLU-style data
                for subject, subject_data in task.data.items():
                    for i in range(len(subject_data["questions"])):
                        example = {
                            "text": subject_data["questions"][i],
                            "target": subject_data["answers"][i]
                        }
                        data["examples"].append(example)
            else:
                # Handle VQA-style data
                for example in task.data:
                    data["examples"].append({
                        "question": example["question"],
                        "image_path": example["image_path"],
                        "target": example["answer"]
                    })
                    
        return data
        
    def _generate_adversarial_data(self, generator: BaseAdversarialGenerator, 
                                 data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adversarial examples"""
        perturbed_data = {"examples": []}
        
        for example in data["examples"]:
            perturbed_example = example.copy()
            
            if "text" in example:
                perturbed_example["text"] = generator.generate_text_perturbation(
                    example["text"]
                )
            elif "question" in example and "image_path" in example:
                perturbed_text, perturbed_images = generator.generate_multimodal_perturbation(
                    example["question"],
                    [example["image_path"]]
                )
                perturbed_example["question"] = perturbed_text
                perturbed_example["image_path"] = perturbed_images[0]
                
            perturbed_data["examples"].append(perturbed_example)
            
        return perturbed_data
        
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across trials"""
        aggregated = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            aggregated[metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }
        return aggregated
        
    def _aggregate_timing(self, timing_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate timing metrics"""
        return {
            "total_time": np.mean([t["total_time"] for t in timing_list]),
            "mean_inference_time": np.mean([t["mean_inference_time"] for t in timing_list]),
            "std_inference_time": np.mean([t["std_inference_time"] for t in timing_list])
        }
        
    def _compute_robustness_metrics(self, clean_metrics: Dict[str, float],
                                  adv_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Compute robustness metrics"""
        robustness = {}
        
        for metric in clean_metrics:
            clean_value = clean_metrics[metric]
            adv_value = adv_metrics[metric]["mean"]
            
            # Relative performance degradation
            robustness[f"{metric}_degradation"] = (clean_value - adv_value) / clean_value
            
            # Absolute performance gap
            robustness[f"{metric}_gap"] = clean_value - adv_value
            
            # Consistency (ratio of adversarial to clean performance)
            robustness[f"{metric}_consistency"] = adv_value / clean_value
            
        return robustness 