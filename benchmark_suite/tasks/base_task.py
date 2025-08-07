from abc import ABC, abstractmethod
from typing import Dict, List, Any
from ..models.base_model import BaseModel

class BaseTask(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = config.get("metrics", ["accuracy"])
        self._image_path_mapping = {}  # For adversarial image mapping
        
    @abstractmethod
    def load_data(self) -> None:
        """Load and prepare the dataset for the task."""
        pass
        
    @abstractmethod
    def evaluate(self, model: BaseModel) -> Dict[str, float]:
        """Run evaluation of the model on this task."""
        pass
        
    @abstractmethod
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format a single example into a prompt for the model."""
        pass
        
    def compute_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Compute all specified metrics for the predictions."""
        results = {}
        for metric_name in self.metrics:
            if metric_name == "accuracy":
                results[metric_name] = sum(p.strip() == t.strip() for p, t in zip(predictions, targets)) / len(targets)
            # Add other metric computations as needed
        return results 
        
    def supports_batch_processing(self) -> bool:
        """Override in child class to enable batch processing."""
        return False

    @abstractmethod
    def get_image_paths(self) -> List[str]:
        """Return a list of all unique local image paths required for the task."""
        pass

    @abstractmethod
    def prepare_batch_data(self) -> List[Dict[str, Any]]:
        """
        Load and structure the entire dataset for batch processing.
        Should return a list of dictionaries, each representing one example.
        """
        pass

    @abstractmethod
    def format_batch_request(self, example: Dict[str, Any], gcs_image_prefix: str) -> Dict[str, Any]:
        """
        Formats a single example from prepare_batch_data into a JSON request
        for the Vertex AI batch prediction API.
        """
        pass

    @abstractmethod
    def evaluate_batch_results(self, batch_results_path: str, metadata_path: str) -> Dict[str, Any]:
        """
        Processes the results from a completed batch job and computes metrics.
        Args:
            batch_results_path: Local path to the downloaded JSONL results from the batch job.
            metadata_path: Local path to the metadata file generated during input creation.
        Returns:
            A dictionary of computed metrics.
        """
        pass
    
    def _update_image_paths(self, adversarial_mapping: Dict[str, str]) -> None:
        """Update task to use adversarial images instead of original ones"""
        self._image_path_mapping = adversarial_mapping
    
    def _get_effective_image_path(self, original_path: str) -> str:
        """Get the effective image path (adversarial if available, original otherwise)"""
        return self._image_path_mapping.get(original_path, original_path)