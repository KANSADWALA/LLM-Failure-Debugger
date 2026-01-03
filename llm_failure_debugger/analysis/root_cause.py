"""
Root cause analysis.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional
from ..type_definitions import FailureSignal, RootCause

# Try to import wikipedia; if not available, fall back gracefully
try:
    import wikipedia
    HAS_WIKIPEDIA = True
except ImportError:
    HAS_WIKIPEDIA = False


# Signal -> Root Cause mapping
SIGNAL_TO_CAUSE = {
    "rag_ungrounded": RootCause.UNGROUNDED_GENERATION,
    "rag_contradiction": RootCause.KNOWLEDGE_CONTRADICTION,
    "logical_contradiction": RootCause.LOGICAL_INCONSISTENCY,
    "nli_contradiction": RootCause.LOGICAL_INCONSISTENCY,
    "semantic_drift": RootCause.SEMANTIC_DRIFT,
    "high_entropy": RootCause.HIGH_UNCERTAINTY,
    "low_self_consistency": RootCause.LOW_SELF_CONSISTENCY,
    "tool_hallucination": RootCause.TOOL_HALLUCINATION,
    "tool_execution_mismatch": RootCause.TOOL_EXECUTION_ERROR,
    "knowledge_contradiction": RootCause.KNOWLEDGE_CONTRADICTION,
    "topic_drift": RootCause.SEMANTIC_DRIFT,
}


class RootCauseAnalyzer:
    """Maps signals to root causes"""
    
    def analyze(self, signals: List[FailureSignal]) -> List[RootCause]:
        """Identify root causes from signals"""
        causes = set()
        
        for signal in signals:
            if signal.name in SIGNAL_TO_CAUSE:
                causes.add(SIGNAL_TO_CAUSE[signal.name])
        
        return list(causes)
    
class KnowledgeVerifier(ABC):
    """
    Verifies factual claims in model output.
    """

    @abstractmethod
    def verify(
        self,
        input_text: str,
        output_text: str,
    ) -> Optional[bool]:
        """
        Returns:
            True  -> factually correct
            False -> factually incorrect
            None  -> cannot verify
        """
        raise NotImplementedError
    

class SimpleKnowledgeVerifier(KnowledgeVerifier):
    """
    Verifies factual claims using Wikipedia summaries.
    """

    def verify(self, input_text: str, output_text: str) -> Optional[bool]:
        """
        Strategy:
        1. Extract entity from question
        2. Fetch Wikipedia summary
        3. Check if answer is supported
        
        Returns None if Wikipedia is not available.
        """
        if not HAS_WIKIPEDIA:
            return None  # Cannot verify without Wikipedia

        try:
            entity = self._extract_entity(input_text)
            if not entity:
                return None

            summary = wikipedia.summary(entity, sentences=2).lower()

            answer_tokens = self._key_tokens(output_text)

            if not answer_tokens:
                return None

            entity_lower = entity.lower()

            # Only keep candidate answer entities (exclude subject & generic words)
            candidate_tokens = [
                t for t in answer_tokens
                if t != entity_lower and t not in {"capital", "country", "city", "state"}
            ]

            # If nothing meaningful to verify, abstain
            if not candidate_tokens:
                return None

            # Special handling for "capital of X" questions
            if "capital" in input_text.lower():
                # Wikipedia summaries usually state "capital is <city>"
                for token in candidate_tokens:
                    if f"capital is {token}" in summary:
                        return True
                return False
            
            # Generic fallback
            for token in candidate_tokens:
                if token in summary:
                    return True

            return False

        except wikipedia.DisambiguationError:
            return None
        except wikipedia.PageError:
            return None
        except Exception:
            return None

    def _extract_entity(self, text: str) -> Optional[str]:
        """
        Very simple heuristic:
        'What is the capital of France?' → 'France'
        """
        match = re.search(r"of\s+([A-Za-z\s]+)\??", text)
        if match:
            return match.group(1).strip()
        return None

    def _key_tokens(self, output_text: str) -> list[str]:
        """
        Extract candidate answers (capitalized words).
        """
        return [
            t.lower()
            for t in re.findall(r"\b[A-Z][a-z]+\b", output_text)
        ]

