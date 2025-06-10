from typing import List
import nltk
from nltk.translate.meteor_score import meteor_score
from .base_metric import BaseMetric

class METEOR(BaseMetric):
    def __init__(self):
        super().__init__()
        nltk.download('wordnet', quiet=True)
        
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            score = meteor_score([ref.split()], pred.split())
            scores.append(score)
        return sum(scores) / len(scores)
