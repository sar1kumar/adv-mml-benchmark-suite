import os
import google.generativeai as genai
from PIL import Image
from typing import Dict, List, Union, Any
from .base_model import BaseModel

class GeminiModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not found in config or environment")
            
        self.model_name = config.get("model_name", "gemini-pro")
        self.max_tokens = config.get("max_tokens", 128000)
        self.temperature = config.get("temperature", 0.95)
        self.model = None
        
    def load_model(self) -> None:
        """Initialize Gemini models"""
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
    def generate(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Generate response(s) for the given prompt(s)"""
        if not self.model:
            self.load_model()
            
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
            
        if isinstance(prompt, list):
            responses = []
            for p in prompt:
                response = self.model.generate_content(
                    p,
                    generation_config=generation_config
                )
                responses.append(response.text)
            return responses
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
        
    def process_image(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Process image and text prompt for VQA tasks"""
        if not self.model:
            self.load_model()
            
        image = Image.open(image_path)
        response = self.model.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )
        answer = response.text
        return {"response": answer}
        
    def process_video(self, video_path: str, prompt: str, **kwargs) -> str:
        """Process video and text prompt for video-based tasks"""
        raise NotImplementedError("Video processing not yet supported for Gemini model") 