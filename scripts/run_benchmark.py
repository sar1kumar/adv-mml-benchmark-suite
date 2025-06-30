#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any

from benchmark_suite.utils.logging import BenchmarkLogger
from benchmark_suite.core.config_parser import ConfigParser
from benchmark_suite.core.model_manager import ModelManager
from benchmark_suite.core.task_manager import TaskManager


def main():
    parser = argparse.ArgumentParser(description="Run multimodal benchmarks")
    parser.add_argument("config", help="Path to benchmark configuration file")
    parser.add_argument("--output", default="results/results.json", help="Path to save results")
    args = parser.parse_args()
    
    # Initialize logger
    logger = BenchmarkLogger(name="benchmark_run")

    # Load and validate configuration using ConfigParser
    try:
        config = ConfigParser.load_and_validate(args.config)
        logger.info(f"Configuration loaded and validated from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Initialize W&B if enabled in config
    wandb_config = config.get("logging", {}).get("wandb", {})
    if wandb_config.get("enabled", False):
        logger.info("W&B logging enabled. Initializing W&B run...")
        try:
            logger.init_wandb(
                config=config, 
                project_name=wandb_config.get("project", "default-benchmark-project"), 
                run_name=wandb_config.get("run_name")
            )
            logger.info(f"W&B run initialized. Project: {wandb_config.get('project', 'default-benchmark-project')}, Run: {wandb_config.get('run_name', 'N/A')}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}. W&B logging will be disabled.")
    else:
        logger.info("W&B logging is disabled in the configuration.")

    try:
        # Initialize models using ModelManager
        models = ModelManager.load_models(config)
        logger.info(f"Initialized models: {list(models.keys())}")
        
        # Initialize tasks using TaskManager
        tasks = TaskManager.load_tasks(config)
        logger.info(f"Initialized tasks: {list(tasks.keys())}")
        
        # Run evaluations
        results = {}
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            model_results = {}
            
            for task_name, task in tasks.items():
                logger.info(f"Running task: {task_name} for model {model_name}")
                try:
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
                        
                except Exception as e:
                    logger.error(f"Error running task {task_name} for model {model_name}: {str(e)}")
                    model_results[task_name] = {"error": str(e)}
                    
            results[model_name] = model_results
        
        # Save results
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

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