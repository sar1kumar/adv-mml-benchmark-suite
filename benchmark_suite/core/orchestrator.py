import os
import json
import time
import logging
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import wandb

from benchmark_suite.core.config_parser import ConfigParser
from benchmark_suite.core.model_manager import ModelManager
from benchmark_suite.core.task_manager import TaskManager
from benchmark_suite.models.base_model import BaseModel
from benchmark_suite.tasks.base_task import BaseTask

class Orchestrator:
    """Coordinates benchmark evaluation across models and tasks"""
    
    def __init__(self, config_path: str):
        self.config = ConfigParser.load_and_validate(config_path)
        self.models = {}
        self.tasks = {}
        self.results = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger("benchmark_suite")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(f"logs/benchmark_{int(time.time())}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def _setup_wandb(self, model_name: str) -> None:
        """Initialize Weights & Biases logging"""
        if "wandb" in self.config["logging"]:
            wandb_config = self.config["logging"]["wandb"]
            wandb.init(
                project=wandb_config["project"],
                entity=wandb_config["entity"],
                name=f"{model_name}_{int(time.time())}",
                config=self.config,
                reinit=True
            )
            
    def _evaluate_task(self, model_name: str, model: BaseModel, 
                      task_name: str, task: BaseTask) -> Dict[str, Any]:
        """Evaluate a single model on a single task"""
        try:
            self.logger.info(f"Starting evaluation of {model_name} on {task_name}")
            start_time = time.time()
            
            results = task.evaluate(model)
            
            elapsed_time = time.time() - start_time
            results["elapsed_time"] = elapsed_time
            
            self.logger.info(f"Completed evaluation of {model_name} on {task_name}")
            self.logger.info(f"Results: {json.dumps(results, indent=2)}")
            
            # Log to W&B if configured
            if "wandb" in self.config["logging"]:
                metrics = {f"{task_name}/{k}": v for k, v in results.items()}
                wandb.log(metrics)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name} on {task_name}: {str(e)}")
            return {"error": str(e)}
            
    def run_evaluations(self, parallel: bool = False) -> Dict[str, Any]:
        """Run all evaluations specified in the config"""
        try:
            # Load models and tasks
            self.models = ModelManager.load_models(self.config)
            self.tasks = TaskManager.load_tasks(self.config)
            
            # Run evaluations for each model
            for model_name, model in self.models.items():
                self.logger.info(f"\nEvaluating model: {model_name}")
                self._setup_wandb(model_name)
                
                model_results = {}
                
                if parallel:
                    # Run tasks in parallel using ThreadPoolExecutor
                    with ThreadPoolExecutor() as executor:
                        future_to_task = {
                            executor.submit(self._evaluate_task, model_name, model, task_name, task): task_name
                            for task_name, task in self.tasks.items()
                        }
                        
                        for future in as_completed(future_to_task):
                            task_name = future_to_task[future]
                            try:
                                model_results[task_name] = future.result()
                            except Exception as e:
                                self.logger.error(f"Task {task_name} generated an exception: {str(e)}")
                                model_results[task_name] = {"error": str(e)}
                else:
                    # Run tasks sequentially
                    for task_name, task in self.tasks.items():
                        model_results[task_name] = self._evaluate_task(model_name, model, task_name, task)
                        
                self.results[model_name] = model_results
                
                if "wandb" in self.config["logging"]:
                    wandb.finish()
                    
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in evaluation pipeline: {str(e)}")
            raise
            
    def save_results(self, output_path: str = None) -> None:
        """Save evaluation results to file"""
        if not output_path:
            output_path = self.config["logging"].get("output_dir", "results")
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, f"benchmark_results_{int(time.time())}.json")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        self.logger.info(f"Results saved to {output_path}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary of the evaluation results"""
        summary = {}
        
        for model_name, model_results in self.results.items():
            model_summary = {
                "tasks_completed": len([t for t in model_results.values() if "error" not in t]),
                "tasks_failed": len([t for t in model_results.values() if "error" in t]),
                "total_time": sum(t.get("elapsed_time", 0) for t in model_results.values() if "elapsed_time" in t),
                "task_metrics": {}
            }
            
            for task_name, task_results in model_results.items():
                if "error" not in task_results:
                    task_metrics = {k: v for k, v in task_results.items() if k != "elapsed_time"}
                    model_summary["task_metrics"][task_name] = task_metrics
                    
            summary[model_name] = model_summary
            
        return summary 