#!/usr/bin/env python3
import os
import yaml
import json
import argparse
from typing import Dict, Any

from benchmark_suite.utils.logging import BenchmarkLogger

from benchmark_suite.models.gemma_model import GemmaModel
from benchmark_suite.models.gemini_model import GeminiModel

from benchmark_suite.tasks.sme_task import SMETask
from benchmark_suite.tasks.vqa_rad import VQARADTask
from benchmark_suite.tasks.embodied_vqa import EmbodiedVQATask
from benchmark_suite.tasks.omni_vqa import OmniMedVQATask
from benchmark_suite.tasks.erqa import ERQATask

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
    if task_name == "sme":
        return SMETask(task_config)
    elif task_name == "omnimed_vqa":
        return OmniMedVQATask(task_config)
    elif task_name == "vqa_rad":
        return VQARADTask(task_config)
    elif task_name == "embodied_vqa":
        return EmbodiedVQATask(task_config)
    elif task_name == "erqa":
        return ERQATask(task_config)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

def main():
    parser = argparse.ArgumentParser(description="Run multimodal benchmarks")
    parser.add_argument("config", help="Path to benchmark configuration file")
    parser.add_argument("--output", default="results/results.json", help="Path to save results")
    args = parser.parse_args()
    
    # Initialize logger
    logger = BenchmarkLogger(name="benchmark_run")

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Initialize W&B if enabled in config
    wandb_config = config.get("logging", {}).get("wandb", {})
    print(wandb_config)
    if wandb_config.get("enabled", False):
        logger.info("W&B logging enabled. Initializing W&B run...")
        try:
            logger.init_wandb(
                config=config, 
                project_name=wandb_config.get("project", "default-benchmark-project"), 
                run_name=wandb_config.get("run_name")
            )
            logger.info(f"W&B run initialized. Project: {wandb_config.get('project', 'default-benchmark-project')}, Run: {logger.get_wandb_run().name if logger.get_wandb_run() else 'N/A'}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}. W&B logging will be disabled.")
    else:
        logger.info("W&B logging is disabled in the configuration.")

    try:
        # Initialize models
        models = {
            model_name: get_model(model_config)
            for model_name, model_config in config["models"].items()
        }
        logger.info(f"Initialized models: {list(models.keys())}")
        
        # Initialize tasks
        tasks = {
            task_name: get_task(task_name, task_config)
            for task_name, task_config in config["tasks"].items()
        }
        logger.info(f"Initialized tasks: {list(tasks.keys())}")
        
        # Run evaluations
        results = {}
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            model_results = {}
            
            for task_name, task in tasks.items():
                logger.info(f"Running task: {task_name} for model {model_name}")
                #try:
                if hasattr(task, 'set_logger'):
                    task.set_logger(logger)

                task_results = task.evaluate(model)
                model_results[task_name] = task_results
                logger.info(f"Results for {task_name} with {model_name}: {json.dumps(task_results, indent=2)}")
                    
                if wandb_config.get("enabled", False) and logger.get_wandb_run():
                    logger.log_summary_metrics(
                        task_name=task_name, 
                        model_name=model_name, 
                        metrics=task_results
                    )
                        
                ##except Exception as e:
                #    logger.error(f"Error running task {task_name} for model {model_name}: {str(e)}")
                #    model_results[task_name] = {"error": str(e)}
                    
            results[model_name] = model_results
        
        output_dir = os.path.dirname(args.output)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        else:
            logger.info(f"Output path '{args.output}' is a filename in the current directory. No subdirectories will be created from path.")

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the benchmark run: {e}", exc_info=True)
    finally:
        if wandb_config.get("enabled", False) and logger.get_wandb_run():
            logger.info("Finishing W&B run...")
            logger.end_wandb_run()
        logger.info("Benchmark run finished.")

if __name__ == "__main__":
    main() 