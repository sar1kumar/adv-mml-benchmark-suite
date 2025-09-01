from typing import List, Dict
import numpy as np
from benchmark_suite.metrics.base_metric import BaseMetric

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
    
    def compute(self, predictions: List[Dict], references: List[Dict], **kwargs) -> float:
        """
        Compute detection metrics for SME task.
        
        Args:
            predictions: List of prediction dictionaries containing model responses
            references: List of reference dictionaries containing 'boxes' field
            
        Returns:
            float: Average IOU score across all valid box matches
        """
        total_ious = []
        
        for pred, ref in zip(predictions, references):
            try:
                # Parse predicted boxes from model_prediction
                pred_boxes = []
                
                # Handle new JSON format
                if isinstance(pred, dict) and isinstance(pred.get("content", {}), dict):
                    content = pred["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        text = content["parts"][0].get("text", "")
                        # Extract boxes from the format "Boxes:\n{BOX0}: [x1, y1, x2, y2]"
                        lines = text.split('\n')
                        for line in lines:
                            if '{BOX' in line and ']' in line:
                                try:
                                    box_str = line[line.index('['): line.index(']')+1]
                                    box = eval(box_str)
                                    if len(box) == 4:  # Ensure valid box format [x1, y1, x2, y2]
                                        pred_boxes.append(box)
                                except:
                                    continue
                else:
                    # Fallback to old format
                    pred_lines = pred.get('model_prediction', '').split('\n')
                    for line in pred_lines:
                        if '{BOX' in line and ']' in line:
                            try:
                                box_str = line[line.index('['): line.index(']')+1]
                                box = eval(box_str)
                                if len(box) == 4:  # Ensure valid box format [x1, y1, x2, y2]
                                    pred_boxes.append(box)
                            except:
                                continue
                
                # Get reference boxes from the nested structure
                ref_boxes = []
                if isinstance(ref.get('boxes', {}), dict):
                    for obj_boxes in ref['boxes'].values():  # Iterate through objects
                        for box_group in obj_boxes:  # Iterate through box groups
                            for box in box_group:  # Get individual boxes
                                if len(box) == 4:  # Ensure valid box format
                                    ref_boxes.append(box)
                
                # Calculate IOUs between predicted and reference boxes
                if pred_boxes and ref_boxes:
                    # For each predicted box, find the best matching reference box
                    for pred_box in pred_boxes:
                        ious = [self._calculate_iou(pred_box, ref_box) for ref_box in ref_boxes]
                        if ious:
                            total_ious.append(max(ious))  # Take the highest IOU for this predicted box
                            
            except Exception as e:
                print(f"Error computing IOU: {e}")
                continue
        
        # Return mean IOU across all valid matches
        return sum(total_ious) / len(total_ious) if total_ious else 0.0

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate areas
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0  # No intersection
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
        
