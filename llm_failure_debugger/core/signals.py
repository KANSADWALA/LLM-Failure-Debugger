"""
Signal extraction modules.
"""

import re
import math
from typing import List
from ..type_definitions import (
    FailureSignal,
    ModelCapabilities,
    ModelInternals,
)
from ..analysis.root_cause import KnowledgeVerifier, SimpleKnowledgeVerifier

# ============================================================
# Base
# ============================================================

class SignalExtractor:
    """Base class for signal extractors"""

    def extract(
        self,
        input_text: str,
        output_text: str,
        internals: ModelInternals,
        capabilities: ModelCapabilities,
    ) -> List[FailureSignal]:
        """Extract signals from input/output pair"""
        if len(input_text.split()) < 2:
            return []
        raise NotImplementedError


# ============================================================
# Entropy / Uncertainty
# ============================================================

class EntropySignalExtractor(SignalExtractor):
    """Detects high uncertainty via token entropy"""

    def extract(self, input_text, output_text, internals, capabilities):
        if not capabilities.supports_logits or not internals.token_probs:
            return []

        entropies = [
            t.get("entropy", 0)
            for t in internals.token_probs
            if "entropy" in t
        ]

        if not entropies:
            return []

        avg = sum(entropies) / len(entropies)

        if len(entropies) < 2:
            return []

        variance = sum((e - avg) ** 2 for e in entropies) / len(entropies)
        std = math.sqrt(variance)

        # High uncertainty = high dispersion
        if std > 0.5:
            return [
                FailureSignal(
                    name="high_entropy",
                    value=min(std / 1.5, 1.0),
                    confidence=0.9,
                    metadata={"std_entropy": std},
                )
            ]
        return []


# ============================================================
# RAG / Grounding
# ============================================================

class RAGGroundingExtractor(SignalExtractor):
    """Detects ungrounded or contradictory RAG claims"""

    def extract(self, input_text, output_text, internals, capabilities):
        if not capabilities.supports_rag or not internals.rag:
            return []

        rag = internals.rag
        signals = []

        if rag.get("supported") is False:
            signals.append(
                FailureSignal(
                    name="rag_contradiction",
                    value=1.0,
                    confidence=0.95,
                    metadata={"evidence": rag.get("evidence")},
                )
            )
        elif rag.get("supported") is None:
            signals.append(
                FailureSignal(
                    name="rag_ungrounded",
                    value=0.6,
                    confidence=0.8,
                    metadata={},
                )
            )

        return signals


class UngroundedClaimExtractor(SignalExtractor):
    """
    Fires when confident factual claims are made without grounding.
    Fallback for non-RAG models (e.g., Ollama).
    """

    FACTUAL_PATTERNS = [
        r"\bwas\b",
        r"\bis\b",
        r"\bwon\b",
        r"\bdiscovered\b",
        r"\binvented\b",
    ]

    def extract(self, input_text, output_text, internals, capabilities):
        if internals.rag is not None:
            return []

        abstention_markers = [
            "i don't have access",
            "i don't know",
            "has not yet been awarded",
            "future events",
        ]

        if any(m in output_text.lower() for m in abstention_markers):
            return []

        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, output_text.lower()):
                return [
                    FailureSignal(
                        name="rag_ungrounded",
                        value=0.6,
                        confidence=0.7,
                        metadata={"reason": "factual_claim_without_grounding"},
                    )
                ]

        return []


# ============================================================
# Reasoning
# ============================================================

class LogicalContradictionExtractor(SignalExtractor):
    """Detects logical contradictions in output"""

    def extract(self, input_text, output_text, internals, capabilities):
        if (
            re.search(r"A\s*>\s*B", output_text)
            and re.search(r"B\s*>\s*C", output_text)
            and re.search(r"C\s*>\s*A", output_text)
        ):
            return [
                FailureSignal(
                    name="logical_contradiction",
                    value=1.0,
                    confidence=0.95,
                    metadata={"type": "transitive_violation"},
                )
            ]
        return []
    
class TopicDriftExtractor(SignalExtractor):
    """
    Detects topic-level drift (not contradictions).
    """

    def _looks_truncated(self, text: str) -> bool:
        text = text.strip()

        if not text:
            return False

        # 1. Ends without sentence punctuation
        if text[-1] not in ".!?":
            return True

        # 2. Unbalanced parentheses or quotes
        if text.count("(") != text.count(")"):
            return True
        if text.count('"') % 2 != 0:
            return True

        # 3. Abrupt ending on conjunctions / continuations
        trailing_tokens = text.lower().split()[-3:]
        if any(t in {"and", "or", "but", "which", "that", "because", "while"} for t in trailing_tokens):
            return True

        # 4. Very low average sentence length (cut mid-thought)
        sentences = re.split(r"[.!?]", text)
        complete = [s for s in sentences if len(s.split()) >= 5]
        if sentences and len(complete) / len(sentences) < 0.5:
            return True

        return False


    def extract(self, input_text, output_text, internals, capabilities):

        # --- Truncation guard (non hard-coded) ---
        if self._looks_truncated(output_text):
            return []

        if input_text.lower() in output_text.lower():
            return []

        if any(char.isdigit() for char in input_text):
            return []
        
        # Topic drift should only fire when the model clearly leaves the domain

        content_keywords = [
            w for w in input_text.lower().split()
            if w not in {"explain", "describe", "what", "is", "the", "a", "an"}
        ]

        # If core topic is mentioned anywhere, do not flag drift
        if any(k in output_text.lower() for k in content_keywords):
            return []

        sentences = re.split(r"[.!?]", output_text)
        drifted = 0
        checked = 0

        for s in sentences:
            tokens = set(s.lower().split())
            if len(tokens) < 8:
                continue

            checked += 1
            if not any(k in s.lower() for k in content_keywords):
                drifted += 1

        if checked >= 3 and drifted / checked >= 0.95:
            return [
                FailureSignal(
                    name="topic_drift",
                    value=1.0,
                    confidence=0.85,
                    metadata={"drifted_sentences": drifted},
                )
            ]

        return []

# ============================================================
# Consistency
# ============================================================

class SelfConsistencyExtractor(SignalExtractor):
    """Detects inconsistent multiple outputs"""

    def extract(self, input_text, output_text, internals, capabilities):
        if not internals.multi_outputs or len(internals.multi_outputs) < 2:
            return []

        unique = set(internals.multi_outputs)
        if len(unique) > 1:
            return [
                FailureSignal(
                    name="low_self_consistency",
                    value=1.0,
                    confidence=0.9,
                    metadata={"variants": list(unique)},
                )
            ]
        return []


# ============================================================
# Tools
# ============================================================

class ToolFailureExtractor(SignalExtractor):
    """Detects hallucinated tools"""

    def extract(self, input_text, output_text, internals, capabilities):
        if not capabilities.supports_tools or not internals.tool_trace:
            return []

        trace = internals.tool_trace
        available = set(trace.get("available_tools", []))
        invoked = set(trace.get("invoked_tools", []))

        if invoked - available:
            return [
                FailureSignal(
                    name="tool_hallucination",
                    value=1.0,
                    confidence=0.95,
                    metadata={"hallucinated_tools": list(invoked - available)},
                )
            ]
        return []


class ToolExecutionErrorExtractor(SignalExtractor):
    """Detects tool execution failures"""

    def extract(self, input_text, output_text, internals, capabilities):
        if not internals.tool_results:
            return []

        for tool, result in internals.tool_results.items():
            if isinstance(result, dict) and "error" in result:
                return [
                    FailureSignal(
                        name="tool_execution_mismatch",
                        value=1.0,
                        confidence=0.95,
                        metadata={"tool": tool, "error": result["error"]},
                    )
                ]
        return []
    
class TemporalHallucinationExtractor(SignalExtractor):
    """Detects claims about future events"""

    def extract(self, input_text, output_text, internals, capabilities):
        abstention_phrases = [
            "i don't know",
            "cannot determine",
            "not yet announced",
            "has not yet been awarded",
            "future event",
        ]

        if any(p in output_text.lower() for p in abstention_phrases):
            # Mark abstention explicitly in internals metadata
            internals.metadata = internals.metadata or {}
            internals.metadata["abstained"] = True
            internals.metadata["suppress_knowledge_checks"] = True
            return []   # ✅ abstention is NOT a failure

        years = re.findall(r"\b(20\d{2})\b", output_text)
        for y in years:
            if int(y) > 2024:
                return [
                    FailureSignal(
                        name="temporal_hallucination",
                        value=1.0,
                        confidence=0.95,
                        metadata={"future_year": y},
                    )
                ]

        return []
    

class IntraAnswerContradictionExtractor(SignalExtractor):
    """Detects contradictions within a single answer"""

    def extract(self, input_text, output_text, internals, capabilities):
        patterns = [
            r"the answer is (\d+)",
            r"might be (\d+)",
        ]

        numbers = []
        for p in patterns:
            numbers += re.findall(p, output_text.lower())

        if len(set(numbers)) > 1:
            return [
                FailureSignal(
                    name="low_self_consistency",
                    value=1.0,
                    confidence=0.9,
                    metadata={"values": numbers},
                )
            ]
        return []
    
class ToolMentionExtractor(SignalExtractor):
    """Detects tool mentions when tools are unsupported"""

    def extract(self, input_text, output_text, internals, capabilities):
        if capabilities.supports_tools:
            return []

        if re.search(r"calling|invoking|using tool", output_text.lower()):
            return [
                FailureSignal(
                    name="tool_hallucination",
                    value=0.8,
                    confidence=0.85,
                    metadata={"text": output_text},
                )
            ]
        return []
    

class KnowledgeContradictionExtractor(SignalExtractor):
    """
    Detects factual contradictions using a knowledge verifier.
    """

    def __init__(self, verifier: KnowledgeVerifier | None = None):
        self.verifier = verifier or SimpleKnowledgeVerifier()

    def extract(self, input_text, output_text, internals, capabilities):

        # Skip verification if model explicitly abstained
        if internals and internals.metadata:
            if internals.metadata.get("abstained"):
                return []

        # Do not verify when model explicitly abstains from future facts
        abstention_markers = [
            "i don't have access",
            "i don't know",
            "has not yet been awarded",
            "future events",
        ]

        # Do not verify when model explicitly abstains
        if any(m in output_text.lower() for m in abstention_markers):
            return []
        
        # Check if suppression flag is set
        if internals and hasattr(internals, 'metadata') and internals.metadata:
            if internals.metadata.get('suppress_knowledge_checks'):
                return []

        # Also check signal metadata
        for existing_signal in getattr(internals, 'signals', []):
            if existing_signal.metadata.get('suppress_knowledge_checks'):
                return []
        
        # Skip verification for simple factual questions with short answers
        if len(input_text.split()) <= 10 and len(output_text.split()) <= 20:
            return []

        result = self.verifier.verify(input_text, output_text)

        if result is False:
            return [
                FailureSignal(
                    name="knowledge_contradiction",
                    value=1.0,
                    confidence=0.95,
                    metadata={"verifier": "wiki"},
                )
            ]

        if result is None:
            return [
                FailureSignal(
                    name="knowledge_contradiction",
                    value=0.4,
                    confidence=0.5,
                    metadata={"verifier": "wiki_unverified"},
                )
            ]

        return []




# ============================================================
# Registry
# ============================================================

EXTRACTORS = [
    EntropySignalExtractor(),
    RAGGroundingExtractor(),
    TemporalHallucinationExtractor(),
    UngroundedClaimExtractor(),
    LogicalContradictionExtractor(),
    TopicDriftExtractor(),
    SelfConsistencyExtractor(),
    IntraAnswerContradictionExtractor(),
    ToolFailureExtractor(),
    ToolExecutionErrorExtractor(),
    ToolMentionExtractor(),
    KnowledgeContradictionExtractor(),  
]


