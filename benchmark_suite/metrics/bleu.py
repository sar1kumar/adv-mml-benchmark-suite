from typing import List
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .base_metric import BaseMetric

class BLEU(BaseMetric):
    def __init__(self, n: int = 4):
        """
        Initialize BLEU-n metric.
        
        Args:
            n: The maximum n-gram order (default: 4 for BLEU-4)
        """
        super().__init__()
        self.n = n
        self.smoothing = SmoothingFunction().method1
        
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower())]
            weights = tuple([1.0/self.n] * self.n)
            score = sentence_bleu(ref_tokens, pred_tokens, 
                                weights=weights,
                                smoothing_function=self.smoothing)
            scores.append(score)
        return sum(scores) / len(scores)
