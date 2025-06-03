from abc import ABC, abstractmethod
from typing import Dict, List, Any
from ..models.base_model import BaseModel

class BaseTask(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = config.get("metrics", ["accuracy"])
        
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