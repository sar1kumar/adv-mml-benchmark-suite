from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any

class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute the metric score.
        
        Args:
            predictions: List of predicted texts/answers
            references: List of reference texts/answers
            **kwargs: Additional metric-specific parameters
            
        Returns:
            float: Metric score
        """
        pass
