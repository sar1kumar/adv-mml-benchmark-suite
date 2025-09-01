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
from benchmark_suite.core.adversarial_manager import AdversarialManager
from benchmark_suite.core.adv_batch_processor import AdvBatchProcessor


def main():
    parser = argparse.ArgumentParser(description="Run multimodal benchmarks with adversarial support")
    parser.add_argument("config", help="Path to benchmark configuration file")
    parser.add_argument("--output", default="results/results.json", help="Path to save results")
    parser.add_argument("--skip-adversarial", action="store_true", 
                       help="Skip adversarial attacks even if configured")
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
        
        # Initialize adversarial manager
        adversarial_manager = None
        if not args.skip_adversarial:
            try:
                adversarial_manager = AdversarialManager(config)
                adversarial_tasks = [name for name in tasks.keys() 
                                   if adversarial_manager.has_adversarial_task(name)]
                if adversarial_tasks:
                    logger.info(f"Adversarial attacks enabled for tasks: {adversarial_tasks}")
                else:
                    logger.info("No adversarial attacks configured")
            except Exception as e:
                logger.warning(f"Failed to initialize adversarial manager: {e}. Continuing without adversarial attacks.")
        else:
            logger.info("Adversarial attacks skipped by user request")
        
        # Determine processing mode
        processing_mode = _determine_processing_mode(models)
        logger.info(f"Detected processing mode: {processing_mode}")
        
        # Run evaluations based on processing mode
        if processing_mode == "batch":
            results = _run_batch_evaluations(models, tasks, config, logger, wandb_config, adversarial_manager)
        else:
            results = _run_realtime_evaluations(models, tasks, config, logger, wandb_config, adversarial_manager)
        
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
                          config: Dict[str, Any], logger, wandb_config: Dict[str, Any],
                          adversarial_manager: AdversarialManager = None) -> Dict[str, Any]:
    """Run evaluations in batch mode with optional adversarial support"""
    logger.info("Starting batch processing workflow")
    
    # Find the batch model
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
    
    results = {}
    model_results = {}
    
    # Process each task
    for task_name, task in tasks.items():
        logger.info(f"Starting batch processing for task: {task_name}")
        
        if not task.supports_batch_processing():
            logger.warning(f"Task {task_name} does not support batch processing. Skipping.")
            model_results[task_name] = {"error": "Task does not support batch processing"}
            continue
        
        try:
            # Check if this task has adversarial attacks enabled
            has_adversarial = (adversarial_manager and 
                             adversarial_manager.has_adversarial_task(task_name))
            
            if has_adversarial:
                logger.info(f"Using AdvBatchProcessor for task: {task_name}")
                batch_processor = AdvBatchProcessor(config, batch_model, adversarial_manager)
            else:
                logger.info(f"Using standard BatchProcessor for task: {task_name}")
                batch_processor = BatchProcessor(config, batch_model)
            
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
                             config: Dict[str, Any], logger, wandb_config: Dict[str, Any],
                             adversarial_manager: AdversarialManager = None) -> Dict[str, Any]:
    """Run evaluations in realtime mode with adversarial support"""
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

                # Handle adversarial attacks for realtime mode
                if adversarial_manager and adversarial_manager.has_adversarial_task(task_name):
                    logger.info(f"Running adversarial evaluation for task: {task_name}")
                    task_results = _run_adversarial_evaluation(task, model, task_name, adversarial_manager, logger)
                else:
                    logger.info(f"Running standard evaluation for task: {task_name}")
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


def _run_adversarial_evaluation(task, model, task_name: str, adversarial_manager: AdversarialManager, logger) -> Dict[str, Any]:
    """Run adversarial evaluation for a single task in realtime mode"""
    try:
        # Load original data
        task.load_data()
        
        # Get image paths from the task
        try:
            original_image_paths = task.get_image_paths()
            logger.info(f"Found {len(original_image_paths)} images for adversarial processing")
        except Exception as e:
            logger.warning(f"Could not get image paths from task: {e}")
            original_image_paths = []
        
        # Generate adversarial images
        adversarial_image_mapping = {}
        if original_image_paths:
            logger.info(f"Generating adversarial images for {len(original_image_paths)} images...")
            adversarial_image_mapping = adversarial_manager.generate_adversarial_images(
                task_name, original_image_paths
            )
        
        # Update task to use adversarial images
        if hasattr(task, '_update_image_paths'):
            task._update_image_paths(adversarial_image_mapping)
        
        # Run evaluation with adversarial data
        results = task.evaluate(model)
        
        # Add adversarial metadata to results
        successful_attacks = sum(1 for orig, adv in adversarial_image_mapping.items() if orig != adv)
        results["adversarial"] = {
            "enabled": True,
            "original_images": len(original_image_paths),
            "adversarial_images": successful_attacks,
            "attack_success_rate": successful_attacks / len(adversarial_image_mapping) if adversarial_image_mapping else 0.0
        }
        
        logger.info(f"Adversarial evaluation completed. Attack success rate: {results['adversarial']['attack_success_rate']:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in adversarial evaluation: {e}")
        # Fallback to standard evaluation
        return task.evaluate(model)


if __name__ == "__main__":
    main() 