from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class BaseAdversarialGenerator(ABC):
    """Base class for generating adversarial examples"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.epsilon = config.get("epsilon", 0.1)  # Perturbation magnitude
        self.num_steps = config.get("num_steps", 10)  # Number of optimization steps
        self.batch_size = config.get("batch_size", 32)
        
    @abstractmethod
    def generate_text_perturbation(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Generate adversarial text examples"""
        pass
        
    @abstractmethod
    def generate_image_perturbation(self, image_path: str) -> str:
        """Generate adversarial image examples"""
        pass
        
    @abstractmethod
    def generate_multimodal_perturbation(self, 
                                       text: Union[str, List[str]], 
                                       image_paths: List[str]) -> tuple:
        """Generate adversarial examples for multimodal inputs"""
        pass
        
    def _validate_perturbation(self, original: Any, perturbed: Any, 
                             modality: str = "text") -> bool:
        """Validate that perturbation meets constraints"""
        if modality == "text":
            # Check semantic similarity
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            orig_emb = model.encode(original)
            pert_emb = model.encode(perturbed)
            
            similarity = np.dot(orig_emb, pert_emb) / (
                np.linalg.norm(orig_emb) * np.linalg.norm(pert_emb)
            )
            return similarity > self.config.get("min_semantic_similarity", 0.7)
            
        elif modality == "image":
            # Check L-inf norm of pixel-wise difference
            
            orig_img = np.array(Image.open(original))
            pert_img = np.array(Image.open(perturbed))
            
            l_inf_norm = np.max(np.abs(orig_img - pert_img))
            return l_inf_norm <= self.epsilon * 255  # Scale epsilon to pixel values
            
        return True 