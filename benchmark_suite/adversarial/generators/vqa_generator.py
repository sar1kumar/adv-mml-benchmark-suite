import os
import random
import numpy as np
from typing import Dict, List, Any, Union
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
import torchvision.transforms as T
import torchvision.models as models
from .base_generator import BaseAdversarialGenerator

class VQAAdversarialGenerator(BaseAdversarialGenerator):
    """Generate adversarial examples for VQA-RAD tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.image_size = config.get("image_size", 224)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def generate_text_perturbation(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Generate adversarial questions for VQA"""
        if isinstance(text, list):
            return [self._perturb_question(q) for q in text]
        return self._perturb_question(text)
        
    def _perturb_question(self, question: str) -> str:
        """Apply medical domain-aware perturbations to questions"""
        # Medical term replacements (simplified example)
        medical_terms = {
            "tumor": ["mass", "lesion", "growth"],
            "fracture": ["break", "crack", "discontinuity"],
            "inflammation": ["swelling", "edema", "irritation"],
            "normal": ["healthy", "unremarkable", "typical"],
            "abnormal": ["pathological", "unusual", "atypical"]
        }
        
        words = question.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in medical_terms and random.random() < self.epsilon:
                words[i] = random.choice(medical_terms[word_lower])
                
        perturbed_question = " ".join(words)
        
        if self._validate_perturbation(question, perturbed_question, "text"):
            return perturbed_question
        return question
        
    def generate_image_perturbation(self, image_path: str) -> str:
        """Generate adversarial medical images"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Apply perturbation
        perturbation, p_type = self._generate_medical_image_perturbation(image_tensor)
        perturbed_image = image_tensor + perturbation
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        # Save perturbed image
        output_path = os.path.join(
            os.path.dirname(image_path),
            f"adversarial_{os.path.basename(image_path)}"
        )
        
        perturbed_image = T.ToPILImage()(perturbed_image.squeeze(0))
        perturbed_image.save(output_path)

        if p_type == "metadata":
            self._corrupt_metadata(output_path)
        
        if self._validate_perturbation(image_path, output_path, "image"):
            return output_path
        return image_path
        
    def _generate_medical_image_perturbation(self, image: torch.Tensor) -> tuple:
        """Generate medical domain-specific image perturbations"""
        perturbation = torch.zeros_like(image, device=self.device)

        perturbation_types = [
            "noise",
            "contrast",
            "brightness",
            "text_overlay",
            "metadata",
            "filtered",
            "fgsm",
        ]
        selected_type = random.choice(perturbation_types)
        
        if selected_type == "noise":
            # Add Gaussian noise
            noise = torch.randn_like(image) * self.epsilon
            perturbation += noise
            
        elif selected_type == "contrast":
            # Modify contrast
            mean = torch.mean(image)
            perturbation += (image - mean) * (random.uniform(-self.epsilon, self.epsilon))

        elif selected_type == "brightness":
            # Modify brightness
            perturbation += torch.ones_like(image) * random.uniform(-self.epsilon, self.epsilon)

        elif selected_type == "text_overlay":
            perturbation += self._add_text_overlay(image)

        elif selected_type == "filtered":
            perturbation += self._apply_filter(image)

        elif selected_type == "fgsm":
            perturbation += self._fgsm_attack(image)

        elif selected_type == "metadata":
            pass  # metadata corruption applied after saving

        return perturbation, selected_type

    def _add_text_overlay(self, image: torch.Tensor) -> torch.Tensor:
        """Overlay random text on the image and return the perturbation."""
        pil_img = T.ToPILImage()(image.squeeze(0).cpu())
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size
        text = "TEXT"
        font = ImageFont.load_default()
        x = random.randint(0, max(0, w - 10))
        y = random.randint(0, max(0, h - 10))
        draw.text((x, y), text, fill=(255, 0, 0), font=font)
        new_tensor = T.ToTensor()(pil_img).unsqueeze(0).to(self.device)
        return new_tensor - image

    def _apply_filter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply a simple blur filter to the image."""
        pil_img = T.ToPILImage()(image.squeeze(0).cpu())
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=2))
        new_tensor = T.ToTensor()(pil_img).unsqueeze(0).to(self.device)
        return new_tensor - image

    def _fgsm_attack(self, image: torch.Tensor) -> torch.Tensor:
        """Basic FGSM attack using a randomly initialized model."""
        model = models.resnet18(weights=None).to(self.device)
        model.eval()
        image_adv = image.clone().detach().requires_grad_(True)
        output = model(image_adv)
        target = torch.zeros((1,), dtype=torch.long, device=self.device)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        perturb = self.epsilon * image_adv.grad.sign()
        return perturb.detach()

    def _corrupt_metadata(self, image_path: str) -> None:
        """Corrupt EXIF metadata in the saved image file."""
        img = Image.open(image_path)
        exif = img.getexif()
        for tag in exif:
            exif[tag] = random.randint(0, 255)
        img.save(image_path, exif=exif)
        
    def generate_multimodal_perturbation(self, text: Union[str, List[str]], 
                                       image_paths: List[str]) -> tuple:
        """Generate adversarial examples for both question and image"""
        perturbed_text = self.generate_text_perturbation(text)
        perturbed_images = [self.generate_image_perturbation(img) for img in image_paths]
        return perturbed_text, perturbed_images 
