# llm_failure_debugger/utils/repair.py

from typing import List
from ..type_definitions import RootCause


class PromptRepairEngine:
    """
    Repairs prompts to mitigate known failure root causes.
    """

    def _instruction_for(self, cause: RootCause) -> str | None:

        """ Get instruction text for a given root cause."""

        if cause in {
            RootCause.UNGROUNDED_GENERATION,
            RootCause.KNOWLEDGE_CONTRADICTION,
        }:
            return (
                "Cite sources and say 'I don't know' if unsure."
                "Do not assume the premise is correct. "
                "If the claim cannot be verified, explicitly state that it is false or unknown. "
                "Do not speculate or provide hypothetical alternatives."
            )

        if cause == RootCause.LOGICAL_INCONSISTENCY:
            return (
                "Use formal logic. "
                "Verify whether the conclusion follows from the premises. "
                "If it does not, explicitly state that it does NOT follow. "
                "Do NOT invent conclusions."
            )

        if cause == RootCause.SEMANTIC_DRIFT:
            return "Stay strictly on topic. Do not introduce unrelated concepts."

        if cause == RootCause.HIGH_UNCERTAINTY:
            return "Be concise and conservative. Avoid speculation."

        if cause == RootCause.LOW_SELF_CONSISTENCY:
            return "Generate alternatives internally and return only the best one."

        if cause == RootCause.TOOL_HALLUCINATION:
            return "Only use tools that are explicitly available."

        if cause == RootCause.TOOL_EXECUTION_ERROR:
            return "Include exact tool outputs verbatim."

        return None

    
    def repair(self, prompt: str, causes: List[RootCause]) -> tuple[str, str]:
        """
        Apply prompt transformations based on root causes.

        Returns:
            (repaired_prompt, repair_status)
        """

        # Idempotence: already repaired
        if "Instruction:" in prompt:
            return prompt, "no_change_already_constrained"

        repaired = prompt.strip()
        instructions = []

        for cause in causes:
            instruction = self._instruction_for(cause)
            if instruction and instruction not in instructions:
                instructions.append(instruction)

        if not instructions:
            return None, "no_applicable_repair"

        # Prevent instruction stacking (idempotence)
        if "Instruction:" in prompt:
            return prompt, "no_change_already_constrained"

        repaired_prompt = (
            prompt
            + "\n\nInstruction:\n"
            + "\n".join(f"- {inst}" for inst in instructions)
        )

        return repaired_prompt, "instruction_only"
    
    @staticmethod
    def _append(prompt: str, instruction: str) -> str:
        return f"{prompt}\n\nInstruction: {instruction}"

class RepairEvaluator:
    """
    Evaluates whether a prompt repair led to safe model behavior.
    This is behavior-based, not answer-based.
    """

    SAFE_BEHAVIOR_POLICY = {
        "abstention": [
            "i don't know",
            "cannot provide",
            "no evidence",
            "not enough information",
        ],
        "premise_rejection": [
            "not possible",
            "did not receive",
            "cannot receive",
            "passed away",
            "never happened",
        ],
    }

    @classmethod
    def is_safe(cls, text: str) -> bool:
        text = text.lower()
        for patterns in cls.SAFE_BEHAVIOR_POLICY.values():
            if any(p in text for p in patterns):
                return True
        return False


class AbstentionEvaluator:
    """
    Evaluates whether the model output represents a safe abstention.
    """

    ABSTENTION_MARKERS = [
        "i don't know",
        "cannot provide",
        "not yet announced",
        "no access to",
        "cannot predict",
        "future event",
    ]

    @classmethod
    def is_abstention(cls, text: str) -> bool:
        text = text.lower()
        return any(m in text for m in cls.ABSTENTION_MARKERS)


