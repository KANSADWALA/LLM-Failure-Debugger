"""
Evaluation metrics for debugger performance.
"""

from typing import List, Dict, Set, Tuple
from ..type_definitions import FailureInstance


class EvaluationMetrics:
    """Compute evaluation metrics"""
    
    @staticmethod
    def detection_metrics(
        predicted: List[FailureInstance],
        ground_truth: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 for failure detection.
        
        ground_truth format: [{"failure_type": "hallucination"}, ...]
        """
        pred_types = {f.failure_type.value for f in predicted}
        gt_types = {g["failure_type"] for g in ground_truth}
        
        tp = len(pred_types & gt_types)
        fp = len(pred_types - gt_types)
        fn = len(gt_types - pred_types)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    
    @staticmethod
    def root_cause_f1(
        predicted: List[FailureInstance],
        ground_truth: List[Dict[str, List[str]]]
    ) -> float:
        """
        Compute F1 for root cause attribution.
        
        ground_truth format: [{"root_causes": ["logical_inconsistency"]}, ...]
        """
        tp = fp = fn = 0
        
        for gt in ground_truth:
            gt_causes = set(gt.get("root_causes", []))
            pred_causes = set()
            
            for p in predicted:
                pred_causes.update(p.root_causes)
            
            tp += len(pred_causes & gt_causes)
            fp += len(pred_causes - gt_causes)
            fn += len(gt_causes - pred_causes)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )