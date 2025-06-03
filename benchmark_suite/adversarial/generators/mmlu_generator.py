import random
from typing import Dict, List, Any, Union
import nltk
from nltk.corpus import wordnet
from .base_generator import BaseAdversarialGenerator

class MMLUAdversarialGenerator(BaseAdversarialGenerator):
    """Generate adversarial examples for MMLU tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        nltk.download('wordnet', quiet=True)
        self.perturbation_types = config.get("perturbation_types", ["synonym", "typo", "negation"])
        
    def generate_text_perturbation(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Generate adversarial text by applying controlled perturbations"""
        if isinstance(text, list):
            return [self._perturb_single_text(t) for t in text]
        return self._perturb_single_text(text)
        
    def _perturb_single_text(self, text: str) -> str:
        """Apply perturbation to a single text example"""
        words = text.split()
        num_words = len(words)
        
        # Randomly select words to perturb
        num_perturbations = max(1, int(self.epsilon * num_words))
        indices_to_perturb = random.sample(range(num_words), num_perturbations)
        
        for idx in indices_to_perturb:
            word = words[idx]
            perturbation_type = random.choice(self.perturbation_types)
            
            if perturbation_type == "synonym":
                words[idx] = self._get_synonym(word)
            elif perturbation_type == "typo":
                words[idx] = self._introduce_typo(word)
            elif perturbation_type == "negation":
                words[idx] = self._add_negation(word)
                
        perturbed_text = " ".join(words)
        
        # Validate perturbation
        if self._validate_perturbation(text, perturbed_text, "text"):
            return perturbed_text
        return text  # Return original if perturbation is too strong
        
    def _get_synonym(self, word: str) -> str:
        """Replace word with a synonym"""
        synsets = wordnet.synsets(word)
        if not synsets:
            return word
            
        synonyms = []
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
                    
        return random.choice(synonyms) if synonyms else word
        
    def _introduce_typo(self, word: str) -> str:
        """Introduce a typographical error"""
        if len(word) < 3:
            return word
            
        error_types = ["swap", "delete", "replace", "insert"]
        error_type = random.choice(error_types)
        
        if error_type == "swap":
            idx = random.randint(0, len(word)-2)
            chars = list(word)
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            return "".join(chars)
            
        elif error_type == "delete":
            idx = random.randint(0, len(word)-1)
            return word[:idx] + word[idx+1:]
            
        elif error_type == "replace":
            idx = random.randint(0, len(word)-1)
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            return word[:idx] + char + word[idx+1:]
            
        else:  # insert
            idx = random.randint(0, len(word))
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            return word[:idx] + char + word[idx:]
            
    def _add_negation(self, word: str) -> str:
        """Add or remove negation"""
        negation_words = ["not", "never", "no"]
        if word in negation_words:
            return ""  # Remove negation
        if random.random() < 0.5:  # 50% chance to add negation
            return random.choice(negation_words) + " " + word
        return word
        
    def generate_image_perturbation(self, image_path: str) -> str:
        """Not implemented for MMLU (text-only task)"""
        raise NotImplementedError("MMLU is a text-only task")
        
    def generate_multimodal_perturbation(self, text: Union[str, List[str]], 
                                       image_paths: List[str]) -> tuple:
        """Not implemented for MMLU (text-only task)"""
        raise NotImplementedError("MMLU is a text-only task") 