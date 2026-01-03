# llm_failure_debugger/tracking/active_learning.py

from typing import List
from ..type_definitions import FailureInstance, Severity


class ActiveLearningSelector:
    """
    Selects informative failure instances for human review or retraining.
    
    Strategy:
    - High uncertainty (low confidence)
    - High severity
    - Novel root causes
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        prioritize_severity: bool = True,
    ):
        self.min_confidence = min_confidence
        self.prioritize_severity = prioritize_severity
    
    def select(self, failures: List[FailureInstance], k: int = 10) -> List[FailureInstance]:
        """
        Select top-k failures for labeling.
        """
        if not failures:
            return []
        
        scored = []
        seen_causes = set()
        
        for f in failures:
            score = 0.0
            
            # Prefer uncertain predictions
            if f.confidence < self.min_confidence:
                score += (self.min_confidence - f.confidence)
            
            # Prefer severe failures
            if self.prioritize_severity:
                score += self._severity_weight(f.severity)
            
            # Prefer novel root causes
            novelty = len(set(f.root_causes) - seen_causes)
            score += novelty * 0.5
            
            scored.append((score, f))
            seen_causes.update(f.root_causes)
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:k]]
    
    @staticmethod
    def _severity_weight(severity: Severity) -> float:
        return {
            Severity.CRITICAL: 1.0,
            Severity.HIGH: 0.7,
            Severity.MEDIUM: 0.4,
            Severity.LOW: 0.1,
        }.get(severity, 0.0)


# Backwards-compatibility alias: older code expects `ActiveLearner`.
ActiveLearner = ActiveLearningSelector
