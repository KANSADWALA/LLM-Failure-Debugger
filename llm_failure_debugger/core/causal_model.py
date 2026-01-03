# llm_failure_debugger/core/causal_model.py

from typing import List, Dict
from ..type_definitions import FailureSignal, FailureType, RootCause


class CausalModel:
    """
    Simple structural causal model (SCM) for reasoning about failures.
    
    Graph:
        Signals -> Root Causes -> Failure Types
    """
    
    def __init__(
        self,
        signal_to_cause: Dict[str, RootCause],
        cause_to_failure: Dict[RootCause, List[FailureType]],
    ):
        self.signal_to_cause = signal_to_cause
        self.cause_to_failure = cause_to_failure
    
    def infer_causes(self, signals: List[FailureSignal]) -> List[RootCause]:
        """Infer root causes from signals"""
        causes = set()
        for s in signals:
            if s.name in self.signal_to_cause:
                causes.add(self.signal_to_cause[s.name])
        return list(causes)
    
    def infer_failures(self, causes: List[RootCause]) -> List[FailureType]:
        """Infer failure types from root causes"""
        failures = set()
        for c in causes:
            for f in self.cause_to_failure.get(c, []):
                failures.add(f)
        return list(failures)
    
    def explain(self, signals: List[FailureSignal]) -> Dict[str, List[str]]:
        """
        End-to-end causal explanation.
        """
        causes = self.infer_causes(signals)
        failures = self.infer_failures(causes)
        
        return {
            "signals": [s.name for s in signals],
            "root_causes": [c.value for c in causes],
            "failures": [f.value for f in failures],
        }
