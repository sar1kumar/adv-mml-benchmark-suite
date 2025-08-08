import os
import logging
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from benchmark_suite.adversarial.generators.siglip_ssa_generator import (
    SigLIPEmbeddingGenerator, 
    create_siglip_generator_config
)
from benchmark_suite.adversarial.generators.vqa_generator import VQAAdversarialGenerator
from benchmark_suite.adversarial.generators.base_generator import BaseAdversarialGenerator


class AdversarialManager:
    """Manages adversarial attack generation and coordination"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("benchmark_suite.adversarial")
        self.generators = {}
        self._initialize_generators()
    
    def _initialize_generators(self):
        """Initialize adversarial generators based on config"""
        for task_name, task_config in self.config["tasks"].items():
            if "adversarial" not in task_config or not task_config["adversarial"].get("enabled", False):
                continue
            
            adv_config = task_config["adversarial"]
            generator_type = adv_config["generator_type"]
            
            self.logger.info(f"Initializing {generator_type} generator for task {task_name}")
            
            try:
                if generator_type == "siglip_embedding":
                    generator_config = create_siglip_generator_config(
                        epsilon=adv_config.get("epsilon", 16/255),
                        attack_steps=adv_config.get("attack_steps", 500),
                        attack_step_size=adv_config.get("attack_step_size", 1/255),
                        attack_mode=adv_config.get("attack_mode", "repulsion"),
                        force_cpu=adv_config.get("force_cpu", False),
                        siglip_model_name=adv_config.get("siglip_model_name", "google/siglip-base-patch16-224")
                    )
                    generator = SigLIPEmbeddingGenerator(generator_config)
                    
                elif generator_type == "vqa":
                    generator_config = {
                        "epsilon": adv_config.get("epsilon", 0.1),
                        "num_steps": adv_config.get("attack_steps", 10),
                        "batch_size": adv_config.get("batch_size", 1),
                        "image_size": adv_config.get("image_size", 224)
                    }
                    generator = VQAAdversarialGenerator(generator_config)
                    
                else:
                    raise ValueError(f"Unsupported generator type: {generator_type}")
                
                self.generators[task_name] = generator
                self.logger.info(f"✓ Successfully initialized {generator_type} generator for {task_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {generator_type} generator for {task_name}: {e}")
                raise
    
    def has_adversarial_task(self, task_name: str) -> bool:
        """Check if a task has adversarial attacks enabled"""
        return task_name in self.generators
    
    def generate_adversarial_images(self, task_name: str, image_paths: List[str], 
                             output_dir: str = None) -> Dict[str, str]:
        """Generate adversarial images for a task"""
        if task_name not in self.generators:
            self.logger.warning(f"No adversarial generator found for task {task_name}")
            return {path: path for path in image_paths if isinstance(path, str)}
        
        generator = self.generators[task_name]
        task_config = self.config["tasks"][task_name]
        adv_config = task_config["adversarial"]
        
        # Use provided output_dir or get from config
        output_dir = adv_config.get("adversarial_output_dir", f"adversarial_{task_name}")
        
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Adversarial images will be saved to: {output_dir}")
        
        adversarial_mapping = {}
        
        # Flatten paths
        flat_paths = []
        for path in image_paths:
            if isinstance(path, str):
                flat_paths.append(path)
            elif isinstance(path, list):
                flat_paths.extend([p for p in path if isinstance(p, str)])
        
        self.logger.info(f"Generating adversarial images for {len(flat_paths)} images in task {task_name}")
        
        for i, image_path in enumerate(flat_paths):
            if not os.path.exists(image_path):
                self.logger.warning(f"Image not found: {image_path}")
                adversarial_mapping[image_path] = image_path
                continue
            
            try:
                self.logger.info(f"Processing image {i+1}/{len(flat_paths)}: {os.path.basename(image_path)}")
                
                # Create output path for this specific image
                original_filename = os.path.basename(image_path)
                name, ext = os.path.splitext(original_filename)
                final_adversarial_filename = f"adv_{name}_{task_name}{ext}"
                final_adversarial_path = os.path.join(output_dir, final_adversarial_filename)
                
                # Generate adversarial image with explicit output path
                temp_adversarial_path = generator.generate_image_perturbation(image_path, final_adversarial_path)
                
                if temp_adversarial_path == image_path:
                    # Attack failed, use original
                    adversarial_mapping[image_path] = image_path
                    continue
                
                # If the generator saved it where we wanted, use that path
                if os.path.exists(final_adversarial_path):
                    adversarial_mapping[image_path] = final_adversarial_path
                    self.logger.info(f"✓ Adversarial image saved: {final_adversarial_path}")
                else:
                    # Fallback: generator returned a different path
                    adversarial_mapping[image_path] = temp_adversarial_path
                    self.logger.warning(f"Adversarial image saved to unexpected location: {temp_adversarial_path}")
                    
            except Exception as e:
                self.logger.error(f"Failed to generate adversarial image for {image_path}: {e}")
                adversarial_mapping[image_path] = image_path
        
        successful_attacks = len([orig for orig, adv in adversarial_mapping.items() if orig != adv])
        self.logger.info(f"Generated {successful_attacks} adversarial images out of {len(flat_paths)} total images")
        
        return adversarial_mapping
    
    def generate_adversarial_text(self, task_name: str, texts: List[str]) -> List[str]:
        """Generate adversarial text perturbations"""
        if task_name not in self.generators:
            return texts  # Return original texts if no generator
        
        generator = self.generators[task_name]
        
        try:
            return generator.generate_text_perturbation(texts)
        except Exception as e:
            self.logger.error(f"Failed to generate adversarial text for task {task_name}: {e}")
            return texts  # Fallback to original texts
    
    def generate_multimodal_perturbations(self, task_name: str, texts: List[str], 
                                        image_paths: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Generate both text and image adversarial perturbations"""
        if task_name not in self.generators:
            return texts, {path: path for path in image_paths}
        
        generator = self.generators[task_name]
        
        try:
            perturbed_texts, perturbed_image_paths = generator.generate_multimodal_perturbation(
                texts, image_paths
            )
            
            # Convert list of perturbed image paths to mapping
            image_mapping = {}
            for orig, perturbed in zip(image_paths, perturbed_image_paths):
                image_mapping[orig] = perturbed
                
            return perturbed_texts, image_mapping
            
        except Exception as e:
            self.logger.error(f"Failed to generate multimodal perturbations for task {task_name}: {e}")
            return texts, {path: path for path in image_paths}
    
    def cleanup_adversarial_files(self, task_name: str, file_paths: List[str]):
        """Clean up generated adversarial files"""
        task_config = self.config["tasks"][task_name]
        if "adversarial" not in task_config:
            return
        
        adv_config = task_config["adversarial"]
        if not adv_config.get("cleanup_after_completion", False):
            return
        
        self.logger.info(f"Cleaning up adversarial files for task {task_name}")
        
        cleaned_count = 0
        for file_path in file_paths:
            try:
                if os.path.exists(file_path) and "adversarial" in os.path.basename(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"Removed adversarial file: {file_path}")
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove adversarial file {file_path}: {e}")
        
        self.logger.info(f"Cleaned up {cleaned_count} adversarial files")
    
    def get_adversarial_output_dir(self, task_name: str) -> str:
        """Get the output directory for adversarial images for a task"""
        if task_name not in self.config["tasks"]:
            return f"adversarial_{task_name}"
        
        task_config = self.config["tasks"][task_name]
        if "adversarial" not in task_config:
            return f"adversarial_{task_name}"
        
        adv_config = task_config["adversarial"]
        return adv_config.get("adversarial_output_dir", f"adversarial_{task_name}")
    
    def list_generated_files(self, task_name: str) -> List[str]:
        """List all adversarial files generated for a task"""
        output_dir = self.get_adversarial_output_dir(task_name)
        
        if not os.path.exists(output_dir):
            return []
        
        adversarial_files = []
        for filename in os.listdir(output_dir):
            if filename.startswith("adv_") and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                adversarial_files.append(os.path.join(output_dir, filename))
        
        return adversarial_files 