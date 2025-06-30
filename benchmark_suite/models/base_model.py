from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any

class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and any necessary components."""
        pass
    
    @abstractmethod
    def generate(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Generate response(s) for the given prompt(s)."""
        pass
    
    @abstractmethod
    def process_image(self, image_path: str, prompt: str, **kwargs) -> str:
        """Process image and text prompt for VQA tasks."""
        pass
    
    @abstractmethod
    def process_images(self, audio_path: str, prompt: str, **kwargs) -> str:
        """Process audio and text prompt for audio-based tasks."""
        pass
    
    @abstractmethod
    def process_video(self, video_path: str, prompt: str, **kwargs) -> str:
        """Process video and text prompt for video-based tasks."""
        pass
    
    def __call__(self, *args, **kwargs):
        """Convenience method to call generate()"""
        return self.generate(*args, **kwargs) 