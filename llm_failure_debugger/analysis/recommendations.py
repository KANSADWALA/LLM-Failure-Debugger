"""
Recommendation engine for fixing failures.
"""

from typing import List
from ..type_definitions import RootCause


# Root cause -> Recommended fixes
FIXES = {

    RootCause.KNOWLEDGE_CONTRADICTION: [
        "Encourage the model to explicitly say 'I don't know' for unknown facts",
        "Add temporal grounding instructions (e.g., 'as of 2023')",
        "Use retrieval-augmented generation (RAG) for factual queries",
        "Penalize confident assertions without evidence during training",
    ],
    RootCause.UNGROUNDED_GENERATION: [
        "Add retrieval-augmented generation (RAG)",
        "Force model to cite sources",
        "Enable abstention when uncertain",
    ],
    RootCause.LOGICAL_INCONSISTENCY: [
        "Enforce step-by-step reasoning",
        "Add logical consistency checks",
        "Use formal verification tools",
    ],
    RootCause.SEMANTIC_DRIFT: [
        "Constrain generation scope with instructions",
        "Use intermediate summaries",
        "Apply attention masks to maintain focus",
    ],
    RootCause.HIGH_UNCERTAINTY: [
        "Lower sampling temperature",
        "Trigger human review",
        "Use ensemble methods",
    ],
    RootCause.LOW_SELF_CONSISTENCY: [
        "Apply self-consistency decoding",
        "Generate multiple outputs and vote",
    ],
    RootCause.TOOL_HALLUCINATION: [
        "Restrict tool schema explicitly",
        "Add tool verification layer",
        "Use function calling with strict mode",
    ],
    RootCause.TOOL_EXECUTION_ERROR: [
        "Force verbatim tool output inclusion",
        "Add tool result validation",
    ],
}


class RecommendationEngine:
    """Generates actionable recommendations"""

    def recommend(
        self,
        causes: List[RootCause],
        causal_graph=None,
    ) -> List[str]:
        """Get recommendations for root causes"""

        recommendations = []

        # --------------------------------------------------
        # Rank root causes by causal importance (if graph exists)
        # --------------------------------------------------
        if causal_graph is not None:
            causes = sorted(
                causes,
                key=lambda c: (
                    causal_graph.out_degree(c.value)
                    if causal_graph.has_node(c.value)
                    else 0
                ),
                reverse=True,
            )

        for cause in causes:
            if cause in FIXES:
                recommendations.extend(FIXES[cause])

        # --------------------------------------------------
        # Deduplicate while preserving order
        # --------------------------------------------------
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)

        # --------------------------------------------------
        # Fallback if nothing mapped
        # --------------------------------------------------
        if not unique_recs:
            return [
                "No specific recommendations available for the detected root causes.",
                "Consider adding retrieval, uncertainty handling, or abstention strategies.",
            ]

        return unique_recs

