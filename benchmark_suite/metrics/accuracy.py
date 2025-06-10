from typing import List
from .base_metric import BaseMetric

class Accuracy(BaseMetric):
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute exact match accuracy between predictions and references.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            float: Accuracy score
        """
        correct = sum(1 for p, r in zip(predictions, references) 
                     if p.lower().strip() == r.lower().strip())
        return correct / len(predictions) if predictions else 0
