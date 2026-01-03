"""
Failure inference engine.
"""

import math
from typing import List, Tuple
from ..type_definitions import FailureSignal, FailureType


# Mapping: signal -> failure type with weight
FAILURE_RULES = {
    FailureType.HALLUCINATION: {
        "rag_contradiction": 1.0,
        "rag_ungrounded": 0.6,
        "high_entropy": 0.4,
        "tool_hallucination": 1.0,
        "tool_execution_mismatch": 1.0,
        "knowledge_contradiction": 1.0,
        "temporal_hallucination": 1.0,
    },
    FailureType.REASONING_BREAKDOWN: {
        "logical_contradiction": 0.9,
        "nli_contradiction": 0.9,
        "topic_drift": 0.6,
    },
    FailureType.CONSISTENCY_ERROR: {
        "low_self_consistency": 1.0,
    }
}


class FailureInferenceEngine:
    """Infers failure types from signals"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def infer(
        self, 
        signals: List[FailureSignal]
    ) -> List[Tuple[FailureType, float, List[str]]]:
        """
        Returns: [(failure_type, confidence, contributing_signals), ...]
        """
        results = []
        
        for failure_type, rules in FAILURE_RULES.items():
            score = 0.0
            contributors = set()
            
            for signal in signals:
                if signal.name in rules:
                    weight = rules[signal.name]
                    contribution = signal.value * weight * signal.confidence
                    score += contribution
                    contributors.add(signal.name)
            
            if score >= self.threshold:
                # Cap score to avoid correlated-signal inflation
                capped_score = min(score, 1.5)
                confidence = 1 - math.exp(-capped_score)
                results.append((failure_type, confidence, list(contributors)))

        return results

