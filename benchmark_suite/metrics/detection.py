from typing import List, Dict
import numpy as np
from .base_metric import BaseMetric

class Detection(BaseMetric):
    def __init__(self, iou_threshold: float = 0.5):
        super().__init__()
        self.iou_threshold = iou_threshold
        
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def compute(self, predictions: List[Dict], references: List[Dict], **kwargs) -> Dict[str, float]:
        """
        Compute detection metrics (mAP, precision, recall).
        
        Args:
            predictions: List of dicts containing predicted boxes and classes
            references: List of dicts containing ground truth boxes and classes
            
        Returns:
            Dict containing mAP, precision, and recall scores
        """
        total_precision = []
        total_recall = []
        
        for pred, ref in zip(predictions, references):
            matched = set()
            tp = 0
            
            for pred_box in pred['boxes']:
                best_iou = 0
                best_idx = -1
                
                for i, ref_box in enumerate(ref['boxes']):
                    if i in matched:
                        continue
                        
                    iou = self._compute_iou(pred_box, ref_box)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_idx = i
                
                if best_idx >= 0:
                    tp += 1
                    matched.add(best_idx)
            
            precision = tp / len(pred['boxes']) if pred['boxes'] else 0
            recall = tp / len(ref['boxes']) if ref['boxes'] else 0
            
            total_precision.append(precision)
            total_recall.append(recall)
        
        avg_precision = np.mean(total_precision)
        avg_recall = np.mean(total_recall)
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'mAP': f1,
            'precision': avg_precision,
            'recall': avg_recall
        }
