from typing import List
import spacy
from .base_metric import BaseMetric

class SPICE(BaseMetric):
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')
        
    def _extract_scene_graph(self, text: str):
        doc = self.nlp(text)
        entities = set()
        relations = set()
        
        # Extract entities and their attributes
        for ent in doc.ents:
            entities.add((ent.text, ent.label_))
            
        # Extract relations (dependencies)
        for token in doc:
            if token.dep_ not in ('punct', 'det'):
                relations.add((token.head.text, token.dep_, token.text))
                
        return entities, relations
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_ents, pred_rels = self._extract_scene_graph(pred)
            ref_ents, ref_rels = self._extract_scene_graph(ref)
            
            # F-score for entities
            ent_precision = len(pred_ents & ref_ents) / len(pred_ents) if pred_ents else 0
            ent_recall = len(pred_ents & ref_ents) / len(ref_ents) if ref_ents else 0
            ent_f1 = 2 * (ent_precision * ent_recall) / (ent_precision + ent_recall) if (ent_precision + ent_recall) > 0 else 0
            
            # F-score for relations
            rel_precision = len(pred_rels & ref_rels) / len(pred_rels) if pred_rels else 0
            rel_recall = len(pred_rels & ref_rels) / len(ref_rels) if ref_rels else 0
            rel_f1 = 2 * (rel_precision * rel_recall) / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0
            
            # Combined score
            score = (ent_f1 + rel_f1) / 2
            scores.append(score)
            
        return sum(scores) / len(scores)
