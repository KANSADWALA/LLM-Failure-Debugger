# llm_failure_debugger/core/precheck.py

import re
from datetime import datetime
from typing import Dict
from ..type_definitions import FailureType


class PreOutputFailurePredictor:
    """
    Predicts likely failures BEFORE generation.
    """

    def predict(self, prompt: str) -> Dict[FailureType, float]:
        risks = {}

        if self._future_query(prompt):
            risks[FailureType.HALLUCINATION] = 0.7

        if self._math_or_logic(prompt):
            risks[FailureType.REASONING_BREAKDOWN] = 0.5

        if self._ambiguous(prompt):
            risks[FailureType.CONSISTENCY_ERROR] = 0.4

        return risks

    def _future_query(self, prompt: str) -> bool:
        current_year = datetime.now().year

        years = re.findall(r"\b(20\d{2})\b", prompt)
        for y in years:
            if int(y) > current_year:
                return True

        temporal_phrases = [
            "next year",
            "in the future",
            "upcoming",
            "will be",
            "has not yet",
            "not yet announced",
        ]

        prompt_l = prompt.lower()
        return any(p in prompt_l for p in temporal_phrases)

    def _math_or_logic(self, prompt: str) -> bool:
        return "therefore" in prompt.lower()

    def _ambiguous(self, prompt: str) -> bool:
        return len(prompt.split()) < 6
