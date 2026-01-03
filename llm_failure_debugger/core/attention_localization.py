# llm_failure_debugger/analysis/attention_localization.py

from typing import Dict
from ..type_definitions import FailureSignal


class AttentionStageLocalizer:
    """
    Approximates attention/layer failure by mapping signals to processing stages.
    """

    STAGE_MAP = {
        "rag_ungrounded": "knowledge_retrieval_stage",
        "semantic_drift": "context_integration_stage",
        "logical_contradiction": "reasoning_stage",
        "low_self_consistency": "decoding_stage",
        "high_entropy": "token_selection_stage",
        "tool_hallucination": "tool_planning_stage",
    }

    def localize(self, signals: list[FailureSignal]) -> Dict[str, list[str]]:
        locations = {}

        for s in signals:
            stage = self.STAGE_MAP.get(s.name)
            if stage:
                locations.setdefault(stage, []).append(s.name)

        return locations
