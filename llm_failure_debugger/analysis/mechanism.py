# llm_failure_debugger/analysis/mechanism.py

from typing import Dict
from ..type_definitions import RootCause


class MechanismExplainer:
    """
    Explains failure mechanisms as causal chains.
    """

    MECHANISMS = {
        RootCause.UNGROUNDED_GENERATION: [
            "No grounding context",
            "Model relies on parametric memory",
            "High hallucination risk"
        ],
        RootCause.LOGICAL_INCONSISTENCY: [
            "Chain-of-thought breakdown",
            "Invalid intermediate inference",
            "Contradictory conclusion"
        ],
        RootCause.LOW_SELF_CONSISTENCY: [
            "High decoding variance",
            "Multiple competing continuations",
            "Unstable final answer"
        ],
    }

    def explain(self, causes: list[RootCause]) -> Dict[str, list[str]]:
        return {
            c.value: self.MECHANISMS.get(c, ["Unknown mechanism"])
            for c in causes
        }
