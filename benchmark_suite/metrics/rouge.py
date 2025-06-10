from typing import List
from rouge_score import rouge_scorer
from .base_metric import BaseMetric

class ROUGEL(BaseMetric):
    def __init__(self):
        super().__init__()
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.scorer.score(ref, pred)['rougeL'].fmeasure
            scores.append(score)
        return sum(scores) / len(scores)
