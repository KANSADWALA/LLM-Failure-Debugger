# llm_failure_debugger/training/intervention.py

from typing import Dict
from ..type_definitions import RootCause


class TrainingInterventionPlanner:
    """
    Converts observed failures into training interventions.
    """

    INTERVENTIONS = {
        RootCause.KNOWLEDGE_CONTRADICTION: [
            "Add counterfactual QA pairs involving unknown or future facts",
            "Train with explicit 'I don't know' responses for unanswerable questions",
            "Increase negative sampling for fabricated entities",
            "Introduce fact-verification auxiliary loss during fine-tuning",
        ],
        RootCause.UNGROUNDED_GENERATION: [
            "Add RAG-supervised examples",
            "Add abstention-labeled samples"
        ],
        RootCause.LOGICAL_INCONSISTENCY: [
            "Add step-by-step reasoning data",
            "Add contradiction-labeled samples"
        ],
        RootCause.LOW_SELF_CONSISTENCY: [
            "Add self-consistency training",
            "Reduce temperature during fine-tuning"
        ],
    }

    def plan(self, causes: list[RootCause]) -> Dict[str, list[str]]:
        actions = {}
        for c in causes:
            actions[c.value] = self.INTERVENTIONS.get(c, [])
        return actions
