from typing import List
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from .base_metric import BaseMetric

class CIDEr(BaseMetric):
    def __init__(self, n: int = 4, sigma: float = 6.0):
        super().__init__()
        self.n = n
        self.sigma = sigma
        
    def _compute_tf_idf(self, predictions: List[str], references: List[str]):
        doc_freq = defaultdict(float)
        ngram_freqs = []
        
        # Compute document frequency
        for ref in references:
            tokens = word_tokenize(ref.lower())
            ngrams = set()
            for i in range(self.n):
                for j in range(len(tokens)-i):
                    ngram = tuple(tokens[j:j+i+1])
                    ngrams.add(ngram)
            for ngram in ngrams:
                doc_freq[ngram] += 1
                
        # Compute TF-IDF
        for text in predictions + references:
            tokens = word_tokenize(text.lower())
            freq = defaultdict(float)
            for i in range(self.n):
                for j in range(len(tokens)-i):
                    ngram = tuple(tokens[j:j+i+1])
                    freq[ngram] += 1
            for ngram, count in freq.items():
                freq[ngram] = count * np.log(len(references) / doc_freq[ngram])
            ngram_freqs.append(freq)
            
        return ngram_freqs
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        ngram_freqs = self._compute_tf_idf(predictions, references)
        scores = []
        
        for i, pred_freq in enumerate(ngram_freqs[:len(predictions)]):
            ref_freqs = ngram_freqs[len(predictions):]
            score = 0
            for ref_freq in ref_freqs:
                num = sum(pred_freq[ng] * ref_freq[ng] for ng in pred_freq if ng in ref_freq)
                denom = np.sqrt(sum(f**2 for f in pred_freq.values()) * 
                              sum(f**2 for f in ref_freq.values()))
                if denom > 0:
                    score += num / denom
            scores.append(score / len(ref_freqs))
            
        return sum(scores) / len(scores)
