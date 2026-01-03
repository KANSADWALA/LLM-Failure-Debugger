"""
Shared type definitions for the LLM debugger framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import time


class FailureType(Enum):
    """High-level failure categories"""
    HALLUCINATION = "hallucination"
    REASONING_BREAKDOWN = "reasoning_breakdown"
    CONSISTENCY_ERROR = "consistency_error"


class Severity(Enum):
    """Failure severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RootCause(Enum):
    """Root causes of failures"""
    UNGROUNDED_GENERATION = "ungrounded_generation"
    KNOWLEDGE_CONTRADICTION = "knowledge_contradiction"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"
    SEMANTIC_DRIFT = "semantic_drift"
    HIGH_UNCERTAINTY = "high_uncertainty"
    LOW_SELF_CONSISTENCY = "low_self_consistency"
    TOOL_HALLUCINATION = "tool_hallucination"
    TOOL_EXECUTION_ERROR = "tool_execution_error"

@dataclass
class FailureLocalization:
    """
    Explicit localization of an LLM failure based on observed signals
    and internal evidence.
    """
    level: str            # e.g. "reasoning", "grounding", "generation"
    evidence: List[str]   # signal names or internal identifiers
    confidence: float

@dataclass
class FailurePrediction:
    """
    Prediction of a likely failure based on learned causal structure.
    """
    failure_type: str
    probability: float
    based_on: List[str]   # causal parent nodes

@dataclass(frozen=True)
class FailureSignal:
    """Atomic evidence of a problem"""
    name: str
    value: float  # [0, 1]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureInstance:
    """A diagnosed failure"""
    failure_id: str
    failure_type: FailureType
    severity: Severity
    confidence: float
    explanation: str
    signals: List[str]  # signal names contributing to this failure
    root_causes: List[str]
    recommendations: List[str]
    input_text: str
    output_text: str
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)
    localization: Optional[FailureLocalization] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model": {
                "provider": self.metadata.get("provider"),
                "name": self.metadata.get("model"),
            },
            "prompt": self.input_text,
            "output": self.output_text,
            "failure": {
                "id": self.failure_id,
                "type": self.failure_type.value,
                "severity": self.severity.value,
                "confidence": self.confidence,
                "signals": self.signals,
                "root_causes": self.root_causes,
                "explanation": self.explanation,
            },
            "localization": (
                {
                    "level": self.localization.level,
                    "evidence": self.localization.evidence,
                    "confidence": self.localization.confidence,
                }
                if self.localization else None
            ),
            "recommendations": self.recommendations,
        }
    
    def pretty_print(self):
        print("\n------------------------------")
        print(f"Failure Type : {self.failure_type.value}")
        print(f"Severity     : {self.severity.value}")
        print(f"Confidence   : {self.confidence:.3f}")

        if self.signals:
            print("Signals:")
            for s in self.signals:
                print(f"  - {s}")

        if self.root_causes:
            print("Root Causes:")
            for rc in self.root_causes:
                print(f"  - {rc}")

        if self.recommendations:
            print("Recommendations:")
            for rec in self.recommendations:
                print(f"  - {rec}")

        interventions = self.metadata.get("training_interventions")
        if interventions:
            print("Training Interventions:")
            for cause, actions in interventions.items():
                print(f"  {cause}:")
                for action in actions:
                    print(f"    - {action}")



@dataclass
class ModelInternals:
    """Container for model internal signals"""
    token_probs: Optional[List[Dict[str, Any]]] = None
    rag: Optional[Dict[str, Any]] = None
    nli_pairs: Optional[List[Dict[str, Any]]] = None
    tool_trace: Optional[Dict[str, Any]] = None
    tool_results: Optional[Dict[str, Any]] = None
    multi_outputs: Optional[List[str]] = None
    output_embeddings: Optional[List[List[float]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() if v is not None
        }


@dataclass
class ModelCapabilities:
    """Declares model capabilities"""
    supports_logits: bool = False
    supports_embeddings: bool = False
    supports_nli: bool = False
    supports_rag: bool = False
    supports_tools: bool = False
    supports_multi_output: bool = False
    
@dataclass
class DebugResult:
    """Result of debugging analysis"""
    input_text: str
    output_text: str
    failures: List[FailureInstance]
    signals: List[FailureSignal]
    metadata: Dict[str, Any] = field(default_factory=dict)
    has_failures: bool = field(init=False)

    
    def __post_init__(self):
        self.has_failures = len(self.failures) > 0

    def __str__(self):
        """String representation that includes failure log path at the end"""
        output = []
        
        if self.has_failures:
            output.append("\n🚨 FAILURES DETECTED")
            
            for failure in self.failures:
                output.append("\n------------------------------")
                output.append(f"Failure Type : {failure.failure_type.value}")
                output.append(f"Severity     : {failure.severity.value}")
                output.append(f"Confidence   : {failure.confidence:.2f}")
                
                if failure.signals:
                    output.append("Signals:")
                    for s in failure.signals:
                        output.append(f"  - {s}")
                
                if failure.root_causes:
                    output.append("Root Causes:")
                    for rc in failure.root_causes:
                        output.append(f"  - {rc}")
                
                if failure.recommendations:
                    output.append("Recommendations:")
                    for rec in failure.recommendations:
                        output.append(f"  - {rec}")
                
                if failure.localization:
                    output.append("Localization:")
                    output.append(f"  Level      : {failure.localization.level}")
                    output.append(f"  Evidence   : {failure.localization.evidence}")
                    output.append(f"  Confidence : {failure.localization.confidence:.2f}")
            
            # Add log path at the end
            if self.metadata and self.metadata.get("failure_log_path"):
                output.append(f"\n🧾 Failure log saved to: {self.metadata['failure_log_path']}")
        else:
            output.append("\n✅ No failures detected.")
        
        # Add repaired prompt if available
        if self.metadata and self.metadata.get("repaired_prompt"):
            output.append("\n🔧 REPAIRED PROMPT SUGGESTED")
            output.append("-" * 80)
            output.append(self.metadata["repaired_prompt"])
        
        return "\n".join(output)
    
    def get_by_severity(self, severity: Severity) -> List[FailureInstance]:
        """Filter failures by severity"""
        return [f for f in self.failures if f.severity == severity]
    
    def get_by_type(self, failure_type: FailureType) -> List[FailureInstance]:
        """Filter failures by type"""
        return [f for f in self.failures if f.failure_type == failure_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "has_failures": self.has_failures,
            "failures": [f.to_dict() for f in self.failures],
            "signals": [
                {"name": s.name, "value": s.value, "confidence": s.confidence, "metadata": s.metadata}
                for s in self.signals
            ],
        }
    
    def to_markdown(self) -> str:
        lines = ["## Model Output\n", self.output_text + "\n"]

        if not self.has_failures:
            lines.append("✅ No failures detected.")
            return "\n".join(lines)

        lines.append("\n## Failures Detected\n")
        for f in self.failures:
            lines.append(f"### {f.failure_type.value}")
            lines.append(f"- Severity: {f.severity.value}")
            lines.append(f"- Confidence: {f.confidence:.3f}")

            if f.root_causes:
                lines.append("- Root Causes:")
                for rc in f.root_causes:
                    lines.append(f"  - {rc}")

            if f.recommendations:
                lines.append("- Recommendations:")
                for rec in f.recommendations:
                    lines.append(f"  - {rec}")

        return "\n".join(lines)
    
    def pretty_print(self):
        print("\n==============================")
        print("MODEL OUTPUT")
        print("==============================")
        print(self.output_text)

        if not self.has_failures:
            print("\n✅ No failures detected.")
            return

        print("\n🚨 FAILURES DETECTED")
        for failure in self.failures:
            failure.pretty_print()
        
    @property
    def recommendations(self) -> List[str]:
        """
        Aggregated recommendations across all failures.
        """
        recs = []
        for f in self.failures:
            recs.extend(f.recommendations)
        return list(dict.fromkeys(recs))


    @property
    def repaired_prompt(self) -> Optional[str]:
        """
        Suggested repaired prompt, if available.
        """
        return self.metadata.get("repaired_prompt") if hasattr(self, "metadata") else None

