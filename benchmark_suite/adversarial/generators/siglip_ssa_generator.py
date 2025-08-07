import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from typing import Dict, List, Any, Union, Callable
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from transformers import SiglipVisionModel, SiglipImageProcessor
from tqdm import tqdm
from abc import abstractmethod
from math import ceil

from .base_generator import BaseAdversarialGenerator



def get_device():
    """Get the best available device with fallback logic"""
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works
            test_tensor = torch.tensor([1.0], device='cuda')
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            return device
        except Exception as e:
            print(f"CUDA available but not working: {e}")
            print("Falling back to CPU")
            return torch.device('cpu')
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')


# ========================= DCT Functions =========================
def dct1(x):
    """Discrete Cosine Transform, Type I"""
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """The inverse of DCT-I, which is just a scaled DCT-I"""
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """Discrete Cosine Transform, Type II (a.k.a. the DCT)"""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    """The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III"""
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """2-dimentional Discrete Cosine Transform, Type II"""
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """The inverse to 2D DCT-II"""
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def clamp(x: torch.tensor, min_value: float = 0, max_value: float = 1):
    return torch.clamp(x, min=min_value, max=max_value)


# ========================= Base Attacker =========================
class AdversarialInputAttacker:
    def __init__(self, model: List[torch.nn.Module],
                 epsilon=16 / 255,
                 norm='Linf',
                 force_cpu: bool = False):
        assert norm in ['Linf', 'L2']
        self.norm = norm
        self.epsilon = epsilon
        self.models = model
        self.force_cpu = force_cpu
        self.init()
        self.model_distribute()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n = len(self.models)

    @abstractmethod
    def attack(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)

    def model_distribute(self):
        """Make each model on appropriate device"""
        if self.force_cpu:
            for model in self.models:
                model.to(torch.device('cpu'))
                model.device = torch.device('cpu')
            return
            
        device = get_device()
        
        if device.type == 'cuda':
            num_gpus = torch.cuda.device_count()
            models_each_gpu = ceil(len(self.models) / num_gpus)
            for i, model in enumerate(self.models):
                try:
                    device_id = min(num_gpus - 1, i // models_each_gpu)
                    model.to(torch.device(f'cuda:{device_id}'))
                    model.device = torch.device(f'cuda:{device_id}')
                except Exception as e:
                    print(f"Failed to move model to GPU, using CPU: {e}")
                    model.to(torch.device('cpu'))
                    model.device = torch.device('cpu')
        else:
            for model in self.models:
                model.to(device)
                model.device = device

    def init(self):
        """Set the model parameters requires_grad to False"""
        for model in self.models:
            model.requires_grad_(False)
            model.eval()

    def to(self, device: torch.device):
        for model in self.models:
            try:
                model.to(device)
                model.device = device
            except Exception as e:
                print(f"Failed to move model to {device}, keeping on current device: {e}")
        self.device = device

    def clamp(self, x: torch.Tensor, ori_x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.norm == 'Linf':
            x = torch.clamp(x, min=ori_x - self.epsilon, max=ori_x + self.epsilon)
        elif self.norm == 'L2':
            difference = x - ori_x
            distance = torch.norm(difference.view(B, -1), p=2, dim=1)
            mask = distance > self.epsilon
            if torch.sum(mask) > 0:
                difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * self.epsilon
                x = ori_x + difference
        x = torch.clamp(x, min=0, max=1)
        return x


# ========================= Feature Extractors =========================
class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SigLIPFeatureExtractor(BaseFeatureExtractor):
    """Surrogate wrapper around SigLIP vision encoder"""
    
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", force_cpu: bool = False):
        super(SigLIPFeatureExtractor, self).__init__()
        # SigLIP default preprocessing: resize -> normalize(mean=0.5,std=0.5)
        self.normalizer = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        print(f"Loading SigLIP model: {model_name}")
        try:
            # Load HF SigLIP image processor and vision model
            self.processor = SiglipImageProcessor.from_pretrained(model_name)
            self.model = SiglipVisionModel.from_pretrained(model_name)
            print("✓ SigLIP model loaded successfully")
        except Exception as e:
            print(f"Error loading SigLIP model: {e}")
            raise
        
        self.device = torch.device('cpu') if force_cpu else get_device()
        self.force_cpu = force_cpu
        
        try:
            # Move model to device, set eval and disable grads
            self.model.to(self.device)
            self.model.eval().requires_grad_(False)
            print(f"✓ SigLIP model moved to {self.device}")
        except Exception as e:
            print(f"Failed to move model to {self.device}, using CPU: {e}")
            self.device = torch.device('cpu')
            self.model.to(self.device)
            self.model.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: batch of images as float tensors in [0,1], shape (B,3,H,W)
        returns: pooled embeddings, shape (B, hidden_size)
        """
        if x.ndim == 2:
            return x
        # Clamp to valid range
        x = torch.clamp(x, min=0.0, max=1.0)
        # Apply resize + normalize, then move to device
        try:
            pixel_values = self.normalizer(x).to(self.device)
            # Forward through SigLIP vision encoder
            outputs = self.model(pixel_values=pixel_values)
            # pooled_output is the (B, hidden_size) tensor
            return outputs.pooler_output
        except Exception as e:
            print(f"Error in SigLIP forward pass: {e}")
            # Return a dummy embedding if forward fails
            return torch.zeros(x.shape[0], 768).to(x.device)


# ========================= Loss Functions =========================
class TargetedFeatureLoss:
    """Loss function for targeted feature attacks"""
    
    def __init__(self, models, idx_fn, feature_loss=None):
        self.models = models
        self.idx_fn = idx_fn
        self.feature_loss = feature_loss or nn.MSELoss()
        self.count = 0
        
        # Placeholders — you'll call one of these before attacking
        self._gt_embeddings = None  # list of tensors
        self._target_embedding = None  # single tensor

    def set_ground_truth(self, x_nat: torch.Tensor):
        """Encode x_nat under each surrogate and store its embeddings."""
        with torch.no_grad():
            try:
                self._gt_embeddings = []
                for m in self.models:
                    x_device = x_nat.to(m.device)
                    emb = m(x_device).to(x_nat.device)
                    self._gt_embeddings.append(emb)
            except Exception as e:
                print(f"Error setting ground truth: {e}")
                # Create dummy embeddings
                self._gt_embeddings = [torch.zeros(x_nat.shape[0], 768).to(x_nat.device) for _ in self.models]
        self._target_embedding = None
        self.count = 0

    def set_target_embedding(self, emb: torch.Tensor):
        """Store a single target embedding and switch into targeted mode."""
        # expect emb shape (1, D)
        self._target_embedding = emb.detach().clone()
        self._gt_embeddings = None
        self.count = 0

    def __call__(self, x_adv: torch.Tensor, _):
        """Called by SSA_CommonWeakness at each PGD step."""
        try:
            idx = self.idx_fn(self.count)
            model = self.models[idx]
            feat = model(x_adv.to(model.device)).to(x_adv.device)

            if self._target_embedding is not None:
                # Pull toward the fixed target
                loss = self.feature_loss(feat, self._target_embedding.to(feat.device))
            else:
                # Push away from the stored ground-truth embedding
                loss = self.feature_loss(feat, self._gt_embeddings[idx])

            self.count += 1
            return loss
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            # Return a dummy loss
            return torch.tensor(0.0, requires_grad=True).to(x_adv.device)


# ========================= SSA Attack =========================
class SSA_CommonWeakness(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 outer_optimizer=None,
                 reverse_step_size=16 / 255 / 15,
                 inner_step_size: float = 250,
                 force_cpu: bool = False,
                 *args,
                 **kwargs):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        self.reverse_step_size = reverse_step_size
        super(SSA_CommonWeakness, self).__init__(model, *args, **kwargs)
        self.inner_step_size = inner_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in tqdm(range(self.total_step), desc="SSA Attack"):
            x.grad = None
            self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                grad = self.get_grad(x, y, model)
                self.grad_record.append(grad)
                x.requires_grad = False
                # Update
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / (
                            torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                                N, 1, 1, 1) + 1e-5)
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / (
                            torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                                N, 1, 1, 1) + 1e-5)
                    x += self.inner_step_size * inner_momentum
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
        patch = now
        if self.outer_optimizer is None:
            fake_grad = (patch - self.original)
            self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
            patch.mul_(0)
            patch.add_(self.original)
            patch.add_(ksi * self.outer_momentum.sign())
        else:
            fake_grad = - ksi * (patch - self.original)
            self.outer_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original)
            patch.grad = fake_grad
            self.outer_optimizer.step()
        patch = clamp(patch)
        del self.grad_record
        del self.original
        return patch

    def get_grad(self, x, y, model):
        rho = 0.5
        N = 20
        sigma = 16
        noise = 0
        for n in range(N):
            x.requires_grad = True
            gauss = torch.randn(*x.shape) * (sigma / 255)
            gauss = gauss.to(x.device)
            x_dct = dct_2d(x + gauss).to(x.device)
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).to(x.device)
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad=True)
            logit = model(x_idct.to(model.device)).to(x_idct.device)
            loss = self.criterion(logit, y)
            loss.backward()
            x.requires_grad = False
            noise += x_idct.grad.data
            x.grad = None
        noise = noise / N
        return noise


# ========================= Main Generator =========================
class SigLIPEmbeddingGenerator(BaseAdversarialGenerator):
    """Generate adversarial examples using SigLIP embedding attacks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.image_size = config.get("image_size", 224)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.siglip_model_name = config.get("siglip_model_name", "google/siglip-base-patch16-224")
        self.attack_steps = config.get("attack_steps", 500)
        self.attack_step_size = config.get("attack_step_size", 1/255)
        self.attack_mode = config.get("attack_mode", "repulsion")  # "repulsion" or "attraction"
        
        # Initialize SigLIP feature extractor
        self.siglip = SigLIPFeatureExtractor(self.siglip_model_name)
        
        # Target options for confusing images
        self.target_options = [
            "Answer is option: E",
            "The correct answer is E",
            "Choose option E"
        ]
        
    def _validate_perturbation(self, original: Any, perturbed: Any, 
                             modality: str = "text") -> bool:
        """Override base validation to handle image size differences"""
        if modality == "text":
            # Use parent class validation for text
            return super()._validate_perturbation(original, perturbed, modality)
            
        elif modality == "image":
            # Custom validation for images that handles size differences
            try:
                orig_img = np.array(Image.open(original))
                pert_img = np.array(Image.open(perturbed))
                
                # Resize both images to the same size for comparison
                target_size = (self.image_size, self.image_size)
                orig_resized = np.array(Image.fromarray(orig_img).resize(target_size))
                pert_resized = np.array(Image.fromarray(pert_img).resize(target_size))
                
                # Check L-inf norm of pixel-wise difference on resized images
                l_inf_norm = np.max(np.abs(orig_resized.astype(float) - pert_resized.astype(float)))
                is_valid = l_inf_norm <= self.epsilon * 255  # Scale epsilon to pixel values
                
                return is_valid
                
            except Exception as e:
                print(f"Warning: Image validation failed: {e}")
                return True  # Accept the perturbation if validation fails
                
        return True
        
    def generate_text_perturbation(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Generate adversarial text by injecting confusing answer hints"""
        if isinstance(text, list):
            return [self._perturb_text_with_answer_hints(t) for t in text]
        return self._perturb_text_with_answer_hints(text)
    
    def _perturb_text_with_answer_hints(self, text: str) -> str:
        """Add subtle answer hints or confusing elements to text"""
        # Add confusing answer hints at random positions
        if random.random() < self.epsilon:
            hint = random.choice(self.target_options)
            insertion_point = random.randint(0, len(text))
            perturbed = text[:insertion_point] + " " + hint + " " + text[insertion_point:]
            
            if self._validate_perturbation(text, perturbed, "text"):
                return perturbed
        return text
    
    def generate_image_perturbation(self, image_path: str, output_path: str = None) -> str:
        """Generate adversarial image using SigLIP embedding attack"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            print(f"Processing image: {image_path}")
            
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            
            # Create target image if in attraction mode
            target_embedding = None
            if self.attack_mode == "attraction":
                print("Creating target embedding...")
                target_embedding = self._create_target_embedding()
            
            # Perform SSA attack
            print("Starting SSA attack...")
            adversarial_image = self._siglip_embedding_attack(image, target_embedding)
            
            # Determine output path
            if output_path is None:
                output_path = os.path.join(
                    os.path.dirname(image_path),
                    f"siglip_adversarial_{os.path.basename(image_path)}"
                )
            
            print(f"Saving adversarial image to: {output_path}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            # Save adversarial image
            self._save_tensor_image(adversarial_image, output_path)
            
            if self._validate_perturbation(image_path, output_path, "image"):
                return output_path
            return image_path
            
        except Exception as e:
            print(f"Error generating image perturbation: {e}")
            return image_path
    
    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for attack"""
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor()
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def _save_tensor_image(self, tensor: torch.Tensor, output_path: str):
        """Save tensor as image"""
        tensor = tensor.detach().cpu().clamp(0, 1).squeeze(0)
        to_pil = T.ToPILImage()
        to_pil(tensor).save(output_path)
    
    def _create_target_image_with_text(self, text: str) -> torch.Tensor:
        """Create a target image with confusing text overlay"""
        # Create a blank white image
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a larger font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.image_size - text_width) // 2
        y = (self.image_size - text_height) // 2
        
        # Draw text in red
        draw.text((x, y), text, fill=(255, 0, 0), font=font)
        
        # Convert to tensor
        transform = T.Compose([T.ToTensor()])
        return transform(img).unsqueeze(0).to(self.device)
    
    def _create_target_embedding(self) -> torch.Tensor:
        """Create target embedding from confusing text image"""
        target_text = random.choice(self.target_options)
        target_image = self._create_target_image_with_text(target_text)
        
        with torch.no_grad():
            target_embedding = self.siglip(target_image)
        
        return target_embedding
    
    def _siglip_embedding_attack(self, x: torch.Tensor, target_embedding: torch.Tensor = None) -> torch.Tensor:
        """Perform SigLIP embedding attack using SSA"""
        models = [self.siglip]
        
        # Index function for single model
        def idx_fn(count, num_models=1, ssa_N=20):
            return 0
        
        # Create loss function
        loss_fn = TargetedFeatureLoss(models, idx_fn, feature_loss=nn.MSELoss())
        
        if target_embedding is not None:
            # Attraction mode: pull toward target
            loss_fn.set_target_embedding(target_embedding)
        else:
            # Repulsion mode: push away from original
            loss_fn.set_ground_truth(x)
        
        # Create attacker
        attacker = SSA_CommonWeakness(
            models,
            epsilon=self.epsilon * 255 / 255,  # Scale epsilon properly
            step_size=self.attack_step_size,
            total_step=self.attack_steps,
            criterion=loss_fn,
            targeted_attack=(target_embedding is not None)
        )
        
        # Perform attack
        adversarial = attacker(x, None)
        return adversarial
    
    def generate_multimodal_perturbation(self, text: Union[str, List[str]], 
                                       image_paths: List[str]) -> tuple:
        """Generate adversarial examples for both text and images"""
        perturbed_text = self.generate_text_perturbation(text)
        perturbed_images = [self.generate_image_perturbation(img) for img in image_paths]
        return perturbed_text, perturbed_images
    
    def generate_target_confusion_image(self, option_letter: str = "E", 
                                      save_path: str = None) -> str:
        """Generate a confusing target image with specific answer option"""
        text = f"Answer is option: {option_letter}"
        target_image = self._create_target_image_with_text(text)
        
        if save_path is None:
            save_path = f"target_confusion_option_{option_letter}.png"
        
        self._save_tensor_image(target_image, save_path)
        return save_path
    
    def attack_with_custom_target(self, image_path: str, target_image_path: str) -> str:
        """Perform targeted attack toward a specific target image"""
        # Load source image
        source_image = self._load_and_preprocess_image(image_path)
        
        # Load target image and get its embedding
        target_image = self._load_and_preprocess_image(target_image_path)
        with torch.no_grad():
            target_embedding = self.siglip(target_image)
        
        # Perform targeted attack
        adversarial_image = self._siglip_embedding_attack(source_image, target_embedding)
        
        # Save result
        output_path = os.path.join(
            os.path.dirname(image_path),
            f"targeted_to_{os.path.splitext(os.path.basename(target_image_path))[0]}_{os.path.basename(image_path)}"
        )
        
        self._save_tensor_image(adversarial_image, output_path)
        return output_path


def create_siglip_generator_config(
    force_cpu: bool = False,
    epsilon: float = 16/255,
    attack_steps: int = 500,
    attack_step_size: float = 1/255,
    attack_mode: str = "repulsion",  # "repulsion" or "attraction"
    image_size: int = 224,
    siglip_model_name: str = "google/siglip-base-patch16-224"
) -> Dict[str, Any]:
    """Create configuration for SigLIP embedding generator"""
    return {
        "epsilon": epsilon,
        "attack_steps": attack_steps,
        "attack_step_size": attack_step_size,
        "attack_mode": attack_mode,
        "image_size": image_size,
        "siglip_model_name": siglip_model_name,
        "force_cpu": force_cpu,
        "num_steps": attack_steps,  # For base class compatibility
        "batch_size": 1,  # For base class compatibility
        "min_semantic_similarity": 0.7  # For text validation
    }