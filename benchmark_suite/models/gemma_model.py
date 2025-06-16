import os
import json
import requests
import base64
from typing import Dict, List, Union, Any
from .base_model import BaseModel

class GemmaModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = config.get("model_name", "gemma3:27b")
        self.max_tokens = config.get("max_tokens", 128000)
        self.temperature = config.get("temperature", 1)
        self.top_k = config.get("top_k", 64)
        self.top_p = config.get("top_p", 0.95)
    def load_model(self) -> None:
        """Verify Ollama is running and model is available"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.model_name, "prompt": "test", "stream": False}
            )
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {str(e)}")
            
    def _call_ollama(self, prompt: str, images: List[str] = None) -> Dict[str, Any]:
        """Make API call to Ollama"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.max_tokens,
                "top_k": self.top_k,
                "top_p": self.top_p
            }
        }
        
        if images:
            payload["images"] = images
        
        response = requests.post(self.ollama_url, json=payload)
        response_json = response.json()

        ollama_eval_duration_ns = response_json.get("eval_duration", 0)
        evaluation_seconds = ollama_eval_duration_ns / 1e9

        return {
            "response": response_json["response"],
            "evaluation_seconds": evaluation_seconds,
            "ollama_metrics": {
                "total_duration_ns": response_json.get("total_duration"),
                "load_duration_ns": response_json.get("load_duration"),
                "prompt_eval_count": response_json.get("prompt_eval_count"),
                "prompt_eval_duration_ns": response_json.get("prompt_eval_duration"),
                "eval_count": response_json.get("eval_count"),
                "eval_duration_ns": ollama_eval_duration_ns
            }
        }
        
    def generate(self, prompt: Union[str, List[str]], **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Generate response(s) for the given prompt(s)"""
        if isinstance(prompt, list):
            return [self._call_ollama(p) for p in prompt]
        return self._call_ollama(prompt)
        
    def process_image(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Process image and text prompt for VQA tasks"""
        # Convert image to base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
            
        # Format prompt for image understanding
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        
        # Pass image in the images array
        return self._call_ollama(formatted_prompt, images=[image_b64])
    def process_images(self, images: List[str], prompt: str, **kwargs) -> Dict[str, Any]:
        """Process multiple images and text prompt for VQA tasks"""
        # Convert images to base64
        image_b64s = [base64.b64encode(image).decode() for image in images]
        
        # Format prompt for image understanding
        formatted_prompt = f"Question: {prompt}\nAnswer:"
        
    def process_video(self, video_path: str, prompt: str, **kwargs) -> str:
        """Process video and text prompt for video-based tasks"""
        raise NotImplementedError("Video processing not yet supported for Gemma model") 