#!/usr/bin/env python3
import os
import sys
import json
import argparse
from typing import Dict, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from benchmark_suite.utils.logging import BenchmarkLogger
from benchmark_suite.core.config_parser import ConfigParser
from benchmark_suite.core.model_manager import ModelManager
from benchmark_suite.core.task_manager import TaskManager
from benchmark_suite.core.batch_processor import BatchProcessor


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
        
        # Determine processing mode
        processing_mode = _determine_processing_mode(models)
        logger.info(f"Detected processing mode: {processing_mode}")
        
        # Run evaluations based on processing mode
        if processing_mode == "batch":
            results = _run_batch_evaluations(models, tasks, config, logger, wandb_config)
        else:
            results = _run_realtime_evaluations(models, tasks, config, logger, wandb_config)
        
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


def _determine_processing_mode(models: Dict[str, Any]) -> str:
    """Determine if we're running in realtime or batch mode"""
    for model in models.values():
        if hasattr(model, 'mode'):
            if model.mode == "batch":
                return "batch"
    return "realtime"


def _run_batch_evaluations(models: Dict[str, Any], tasks: Dict[str, Any], 
                          config: Dict[str, Any], logger, wandb_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluations in batch mode"""
    logger.info("Starting batch processing workflow")
    
    # Find the batch model (should be only one per config validation)
    batch_model = None
    batch_model_name = None
    for model_name, model in models.items():
        if hasattr(model, 'mode') and model.mode == "batch":
            batch_model = model
            batch_model_name = model_name
            break
    
    if not batch_model:
        raise ValueError("No batch model found, but processing mode is batch")
    
    logger.info(f"Using batch model: {batch_model_name}")
    
    # Initialize BatchProcessor
    batch_processor = BatchProcessor(config, batch_model)
    
    results = {}
    model_results = {}
    
    # Process each task
    for task_name, task in tasks.items():
        logger.info(f"Starting batch processing for task: {task_name}")
        
        # Check if task supports batch processing
        if not task.supports_batch_processing():
            logger.warning(f"Task {task_name} does not support batch processing. Skipping.")
            model_results[task_name] = {"error": "Task does not support batch processing"}
            continue
        
        try:
            # Set logger if task supports it
            if hasattr(task, 'set_logger'):
                task.set_logger(logger)
            
            # Run batch evaluation
            task_results = batch_processor.evaluate_task(task_name, task)
            model_results[task_name] = task_results
            
            logger.info(f"Batch results for {task_name}: {json.dumps(task_results, indent=2)}")
            
            # Log to W&B if enabled
            if wandb_config.get("enabled", False) and logger.get_wandb_run():
                logger.log_summary_metrics(
                    task_name=task_name, 
                    model_name=batch_model_name, 
                    metrics=task_results
                )
                
        except Exception as e:
            logger.error(f"Error in batch processing for task {task_name}: {str(e)}")
            model_results[task_name] = {"error": str(e)}
    
    results[batch_model_name] = model_results
    return results


def _run_realtime_evaluations(models: Dict[str, Any], tasks: Dict[str, Any], 
                             config: Dict[str, Any], logger, wandb_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run evaluations in realtime mode (original logic)"""
    logger.info("Starting realtime processing workflow")
    
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
    
    return results


if __name__ == "__main__":
    main() 