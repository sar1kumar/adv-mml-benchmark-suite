#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any

from benchmark_suite.core import ConfigParser, ModelManager, TaskManager
from benchmark_suite.adversarial.generators.mmlu_generator import MMLUAdversarialGenerator
from benchmark_suite.adversarial.generators.vqa_generator import VQAAdversarialGenerator
from benchmark_suite.adversarial.evaluators.robustness_evaluator import RobustnessEvaluator

def get_generator(task_name: str, config: Dict[str, Any]):
    """Get appropriate adversarial generator for task"""
    if task_name == "mmlu":
        return MMLUAdversarialGenerator(config)
    elif task_name in ["vqa_rad", "embodied_vqa"]:
        return VQAAdversarialGenerator(config)
    else:
        raise ValueError(f"No adversarial generator available for task: {task_name}")

def main():
    parser = argparse.ArgumentParser(description="Run adversarial benchmarks")
    parser.add_argument("config", help="Path to benchmark configuration file")
    parser.add_argument("--output", default="results/adversarial",
                       help="Directory to save results")
    parser.add_argument("--num-trials", type=int, default=5,
                       help="Number of adversarial trials per task")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="Perturbation magnitude")
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigParser.load_and_validate(args.config)
    
    # Add adversarial specific config
    adversarial_config = {
        "num_trials": args.num_trials,
        "epsilon": args.epsilon,
        "metrics": ["accuracy", "robustness", "inference_time"]
    }
    
    # Initialize evaluator
    evaluator = RobustnessEvaluator(adversarial_config)
    
    # Load models and tasks
    models = ModelManager.load_models(config)
    tasks = TaskManager.load_tasks(config)
    
    # Run evaluations
    results = {}
    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        model_results = {}
        
        for task_name, task in tasks.items():
            print(f"\nRunning adversarial evaluation on task: {task_name}")
            try:
                # Get appropriate generator for task
                generator = get_generator(task_name, adversarial_config)
                
                # Run evaluation
                task_results = evaluator.evaluate(model, task, generator)
                model_results[task_name] = task_results
                
                print("\nResults summary:")
                print(f"Clean performance: {task_results['clean_performance']}")
                print(f"Adversarial performance: {task_results['adversarial_performance']}")
                print(f"Robustness metrics: {task_results['robustness_metrics']}")
                print(f"Timing metrics: {task_results['timing_metrics']}")
                
            except Exception as e:
                print(f"Error evaluating task {task_name}: {str(e)}")
                model_results[task_name] = {"error": str(e)}
                
        results[model_name] = model_results
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "adversarial_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {output_file}")
    
    # Generate summary
    print("\nOverall Summary:")
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        for task_name, task_results in model_results.items():
            if "error" not in task_results:
                clean_acc = task_results["clean_performance"].get("accuracy", 0)
                adv_acc = task_results["adversarial_performance"]["accuracy"]["mean"]
                rob_deg = task_results["robustness_metrics"]["accuracy_degradation"]
                inf_time = task_results["timing_metrics"]["adversarial"]["mean_inference_time"]
                
                print(f"\nTask: {task_name}")
                print(f"  Clean Accuracy: {clean_acc:.3f}")
                print(f"  Adversarial Accuracy: {adv_acc:.3f}")
                print(f"  Robustness Degradation: {rob_deg:.3f}")
                print(f"  Mean Inference Time: {inf_time:.3f}s")

if __name__ == "__main__":
    main() 