import os
import google.generativeai as genai
from PIL import Image
from typing import Dict, List, Union, Any
import base64
from .base_model import BaseModel

# Add new imports for batch operations
try:
    from google import genai as vertex_genai
    from google.auth import default
except ImportError:
    vertex_genai = None
    default = None


class GeminiModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mode = config.get("mode", "realtime")
        self.model_name = config.get("model_name", "gemini-pro")
        self.max_tokens = config.get("max_tokens", 8192)
        self.temperature = config.get("temperature", 0.95)
        
        if self.mode == "realtime":
            # Realtime mode requires API key
            self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key not found in config or environment for realtime mode")
        elif self.mode == "batch":
            # Batch mode uses gcloud authentication
            self.batch_config = config.get("batch_config", {})
            self.project_id = config.get("project_id")
            self.location = config.get("location", "us-central1")
            if not self.project_id:
                raise ValueError("`project_id` is required for batch mode.")
                
            # Verify gcloud authentication
            try:
                credentials, project = default()
                if not credentials:
                    raise ValueError("No gcloud credentials found. Please run 'gcloud auth application-default login'")
            except Exception as e:
                raise ValueError(f"Failed to authenticate with gcloud: {e}. Please run 'gcloud auth application-default login'")

        self.model = None
        
    def load_model(self) -> None:
        """Initialize Gemini model based on mode"""
        if self.mode == "realtime":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        elif self.mode == "batch":
            # For batch mode, initialize with gcloud credentials
            if not vertex_genai:
                raise ImportError("google-genai library required for batch mode. Install with: pip install google-genai")
            # The vertex_genai client will automatically use ADC
            pass
        
    def is_batch_mode(self) -> bool:
        """Check if model is in batch mode."""
        return self.mode == "batch"
        
    def is_realtime_mode(self) -> bool:
        """Check if model is in realtime mode."""
        return self.mode == "realtime"
        
    def generate(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Generate response(s) - only works in realtime mode"""
        if self.mode != "realtime":
            raise RuntimeError(f"generate() method not supported in {self.mode} mode")
            
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
        """Process image and text prompt for VQA tasks - only works in realtime mode"""
        if self.mode != "realtime":
            raise RuntimeError(f"process_image() method not supported in {self.mode} mode")
            
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

    def process_images(self, images: List[str], prompt: str, **kwargs) -> Dict[str, Any]:
        """Process images and text prompt for VQA tasks - only works in realtime mode"""
        if self.mode != "realtime":
            raise RuntimeError(f"process_images() method not supported in {self.mode} mode")
            
        if not self.model:
            self.load_model()
            
        image_b64s = [Image.open(image_path) for image_path in images]
        response = self.model.generate_content(
            [prompt, *image_b64s],
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
        
    def create_batch_job(self, input_gcs_uri: str, output_gcs_uri_prefix: str) -> str:
        """Create a batch job using the google-genai SDK with gcloud authentication."""
        if not self.is_batch_mode():
            raise RuntimeError("`create_batch_job` can only be called in 'batch' mode.")
            
        if not vertex_genai:
            raise ImportError("google-genai library required for batch mode. Install with: pip install google-genai")
            
        # vertex_genai will automatically use Application Default Credentials
        client = vertex_genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        
        job = client.batches.create(
            model=self.model_name,
            src=input_gcs_uri,
            config={
                "dest": output_gcs_uri_prefix,
            },
        )
        return job.name

    def get_batch_job_status(self, job_name: str) -> Any:
        """Gets the status of a batch job using gcloud authentication."""
        if not self.is_batch_mode():
            raise RuntimeError("`get_batch_job_status` can only be called in 'batch' mode.")
        
        if not vertex_genai:
            raise ImportError("google-genai library required for batch mode. Install with: pip install google-genai")
            
        client = vertex_genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        
        return client.batches.get(name=job_name) 