#!/usr/bin/env python3
import os
import yaml
import json
import argparse
from typing import Dict, Any

from benchmark_suite.utils.logging import BenchmarkLogger

from benchmark_suite.models.gemma_model import GemmaModel
from benchmark_suite.models.gemini_model import GeminiModel

from benchmark_suite.tasks.text_mmlu import MMLUTask
from benchmark_suite.tasks.vqa_rad import VQARADTask
from benchmark_suite.tasks.embodied_vqa import EmbodiedVQATask

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate benchmark configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    required_keys = ["models", "tasks"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config")
            
    return config

def get_model(model_config: Dict[str, Any]):
    """Initialize model based on config"""
    model_type = model_config.get("type", "").lower()
    
    if model_type == "ollama":
        return GemmaModel(model_config)
    elif model_type == "api":
        return GeminiModel(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
def get_task(task_name: str, task_config: Dict[str, Any]):
    """Initialize task based on config"""
    if task_name == "mmlu":
        return MMLUTask(task_config)
    elif task_name == "vqa_rad":
        return VQARADTask(task_config)
    elif task_name == "embodied_vqa":
        return EmbodiedVQATask(task_config)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

def main():
    parser = argparse.ArgumentParser(description="Run multimodal benchmarks")
    parser.add_argument("config", help="Path to benchmark configuration file")
    parser.add_argument("--output", default="results/results.json", help="Path to save results")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize models
    models = {
        model_name: get_model(model_config)
        for model_name, model_config in config["models"].items()
    }
    
    # Initialize tasks
    tasks = {
        task_name: get_task(task_name, task_config)
        for task_name, task_config in config["tasks"].items()
    }
    
    # Run evaluations
    results = {}
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        model_results = {}
        
        for task_name, task in tasks.items():
            print(f"\nRunning task: {task_name}")
            try:
                task_results = task.evaluate(model)
                model_results[task_name] = task_results
                print(f"Results: {json.dumps(task_results, indent=2)}")
            except Exception as e:
                print(f"Error running task {task_name}: {str(e)}")
                model_results[task_name] = {"error": str(e)}
                
        results[model_name] = model_results
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 