"""
Main Debugger class - Public API entry point.
"""

import os
from datetime import datetime
import hashlib
from importlib import metadata
import json
from typing import Optional, List
from unittest import result
from ..adapters.base import LLMAdapter
from ..type_definitions import (
    DebugResult,
    FailureInstance,
    FailureType,
    RootCause,
    Severity,
    ModelInternals,
)
from .signals import EXTRACTORS
from .inference import FailureInferenceEngine
from ..analysis.root_cause import RootCauseAnalyzer
from ..analysis.recommendations import RecommendationEngine
from ..analysis.causal_graph import (
    CausalDatasetBuilder,
    CausalGraphLearner,
    CausalGraphVisualizer, 
    CausalRepairSelector
)
from .precheck import PreOutputFailurePredictor
from ..utils.repair import PromptRepairEngine, AbstentionEvaluator, RepairEvaluator
from ..tracking.tracker import FailureTracker
from ..training.intervention import TrainingInterventionPlanner
from ..type_definitions import FailureLocalization, FailurePrediction


class Debugger:
    """
    Main debugger class for analyzing LLM failures.
    
    This is the primary public API for the library.
    
    Example:
        >>> from llm_failure_debugger.core.debugger import Debugger
        >>> from llm_failure_debugger.adapters.openai_adapter import OpenAIAdapter
        >>> 
        >>> adapter = OpenAIAdapter(api_key="sk-...")
        >>> debugger = Debugger(adapter)
        >>> 
        >>> # Analyze with automatic generation
        >>> result = debugger.debug("What is 2+2?")
        >>> 
        >>> # Or analyze existing output
        >>> result = debugger.analyze("What is 2+2?", "The answer is 5")
        >>> 
        >>> for failure in result.failures:
        >>>     print(f"{failure.failure_type}: {failure.explanation}")
        >>>     print(f"Recommendations: {failure.recommendations}")
    """
    
    def __init__(self, adapter: LLMAdapter, inference_threshold: float = 0.5):
        """
        Initialize the debugger.
        
        Args:
            adapter: LLM adapter for model integration
            inference_threshold: Minimum confidence to report failures
        """
        self.adapter = adapter
        self.extractors = EXTRACTORS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._failure_log_filename = f"failures_log_{timestamp}.jsonl"
        self.inference_engine = FailureInferenceEngine(threshold=inference_threshold)
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.recommendation_engine = RecommendationEngine()
        self.precheck = PreOutputFailurePredictor()
        self.repair_engine = PromptRepairEngine()
        self._causal_failures = []
        self._causal_graph = None
        self._causal_builder = CausalDatasetBuilder()
        self._causal_learner = CausalGraphLearner()
        self.tracker = FailureTracker()
        self.intervention_planner = TrainingInterventionPlanner()


    # --------------------------------------------------
    # Factory constructors (PUBLIC API)
    # --------------------------------------------------

    @classmethod
    def from_openai(cls, api_key: str, **kwargs):
        from ..adapters.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(api_key=api_key, **kwargs)
        return cls(adapter)

    @classmethod
    def from_anthropic(cls, api_key: str, **kwargs):
        from ..adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter(api_key=api_key, **kwargs)
        return cls(adapter)

    @classmethod
    def from_llm(cls, provider: str, model: str | None = None, **kwargs):
        """
        Create a Debugger from a high-level LLM spec.

        Example:
            Debugger.from_llm("ollama", model="llama2")
            Debugger.from_llm("openai", model="gpt-4", api_key="...")
        """
        from ..adapters.factory import create_adapter

        adapter = create_adapter(
            provider=provider,
            model=model,
            **kwargs,
        )
        return cls(adapter)
    
    # --------------------------------------------------
    # Public API methods    
    # --------------------------------------------------
    
    def debug(self, prompt: str, custom_internals: Optional[ModelInternals] = None,
        **generation_kwargs) -> DebugResult:
        """
        Generate output and analyze for failures.
        
        Args:
            prompt: Input prompt
            custom_internals: Optional pre-extracted internals
            **generation_kwargs: Passed to adapter.generate()
        
        Returns:
            DebugResult with failures, signals, and recommendations
        """
        # Optional precheck (not used in analysis but could be logged or acted upon)
        precheck_risks = self.precheck.predict(prompt)

        output, internals = self.adapter.analyze(prompt, **generation_kwargs)

        internals.metadata = internals.metadata or {}
        internals.metadata["precheck_risks"] = {
            k.value: v for k, v in precheck_risks.items()
        }
       
        if custom_internals:
            # Merge custom internals
            for key, value in custom_internals.to_dict().items():
                if value is not None:
                    setattr(internals, key, value)
        
        # result = self.analyze(prompt, output, internals)
        result = self.analyze(prompt, output, internals, 
                              auto_repair=True)

        # Print log path AFTER the result is processed
        if result.metadata.get("_print_log_path_after") and hasattr(self, "_failure_log_path"):
            if not hasattr(self, "_failure_log_path_announced"):
                print(f"\n🧾 Failure log saved to: {self._failure_log_path}")
                self._failure_log_path_announced = True

        # --------------------------------------------------
        # Internal repair effectiveness evaluation (2nd pass)
        # --------------------------------------------------
        if result.metadata.get("repaired_prompt"):
            repaired_output, repaired_internals = self.adapter.analyze(
                result.metadata["repaired_prompt"]
            )

            repaired_result = self.analyze(
                result.metadata["repaired_prompt"],
                repaired_output,
                repaired_internals,
                auto_repair=False,
            )

            result.metadata["post_repair_output"] = repaired_result.output_text
            result.metadata["repair_safe"] = RepairEvaluator.is_safe(
                repaired_result.output_text
            )
        else:
            result.metadata["post_repair_output"] = None
            result.metadata["repair_safe"] = None


        return result
    
    def analyze(self, input_text: str, output_text: str, 
                internals: Optional[ModelInternals] = None, auto_repair: bool = False,) -> DebugResult:
        """
        Analyze existing input/output for failures.
        """
        internals = internals or ModelInternals()
        capabilities = self.adapter.capabilities

        # --------------------------------------------------
        # 1. Extract signals
        # --------------------------------------------------
        signals: List = []
        for extractor in self.extractors:
            extracted = extractor.extract(
                input_text, output_text, internals, capabilities
            )
            signals.extend(extracted)

        # Deduplicate signals (keep strongest, merge metadata)
        unique = {}
        for s in signals:
            if s.name not in unique:
                unique[s.name] = s
            elif s.value > unique[s.name].value:
                merged_metadata = {
                    **(unique[s.name].metadata or {}),
                    **(s.metadata or {}),
                }
                unique[s.name] = s
                unique[s.name].metadata = merged_metadata
        signals = list(unique.values())

        # --------------------------------------------------
        # 2. Infer failures
        # --------------------------------------------------
        failure_hypotheses = self.inference_engine.infer(signals)

        failures: List[FailureInstance] = []

        # --------------------------------------------------
        # 3. Build FailureInstances
        # --------------------------------------------------
        for failure_type, confidence, contributing_signals in failure_hypotheses:
            relevant_signals = [
                s for s in signals if s.name in contributing_signals
            ]

            root_causes = self.root_cause_analyzer.analyze(relevant_signals)
            recommendations = self.recommendation_engine.recommend(
                    root_causes,
                    causal_graph=self._causal_graph,
                )

            # ---- causal root selection (if graph exists) ----
            best_root = None
            if self._causal_graph and root_causes:
                selector = CausalRepairSelector()
                best_root = selector.select_best(root_causes, self._causal_graph)

            failure = FailureInstance(
                failure_id=self._generate_id(input_text, failure_type),
                failure_type=failure_type,
                severity=self._determine_severity(failure_type, confidence),
                confidence=confidence,
                explanation=f"Detected via signals: {', '.join(contributing_signals)}",
                signals=contributing_signals,
                root_causes=[c.value for c in root_causes],
                recommendations=recommendations,
                input_text=input_text,
                output_text=output_text,
                metadata={
                    "all_signals": [s.name for s in signals],
                    "best_root_cause": best_root.value if best_root else None,
                    "training_interventions": self.intervention_planner.plan(root_causes),
                },
            )

            failures.append(failure)

            # ------------------------------
            # Explicit failure localization
            # ------------------------------
            failure.localization = FailureLocalization(
                level=failure.failure_type.value,
                evidence=contributing_signals,
                confidence=confidence,
            )

        # --------------------------------------------------
        # 4. Online causal learning
        # --------------------------------------------------
        if failures:
            self._causal_failures.extend(failures)
            # Learn/update causal graph if enough data
            if len(self._causal_failures) >= 2:
                samples = self._causal_builder.build(self._causal_failures)
                self._causal_graph = self._causal_learner.learn(samples)

        # --------------------------------------------------
        # 5. Optional auto-repair
        # --------------------------------------------------
        repaired_prompt = None
        repair_status = "not_attempted"

        if auto_repair and failures:
            primary_failure = failures[0]

            # 🔒 Confidence & severity gate
            if (
                primary_failure.severity in {Severity.MEDIUM, Severity.HIGH}
                and primary_failure.confidence >= 0.6
            ):
                repaired_prompt, repair_status = self.repair_engine.repair(
                    input_text,
                    [RootCause(primary_failure.root_causes[0])],
                )
            else:
                repair_status = "below_confidence_threshold"

        # --------------------------------------------------
        # 6. Persist failures
        # --------------------------------------------------
        log_path = None

        if failures:
            log_dir = os.path.join(os.getcwd(), "causal_outputs")
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, self._failure_log_filename)
            self._failure_log_path = log_path

            with open(log_path, "a") as f:
                for failure in failures:
                    json.dump(
                        failure.to_dict(),
                        f,
                        indent=2,          # 👈 makes it readable
                        ensure_ascii=False
                    )
                    f.write("\n-------------------------------------------------------------------------\n")        # 👈 separate entries clearly

        # --------------------------------------------------
        # 7. Track failures
        # --------------------------------------------------
        if failures:
            self.tracker.log(
                model_name=self.adapter.__class__.__name__,
                model_version=getattr(self.adapter, "model", "unknown"),
                failures=failures,
            )

        # --------------------------------------------------
        # 8. Return result
        # --------------------------------------------------
        result = DebugResult(
            input_text=input_text,
            output_text=output_text,
            failures=failures,
            signals=signals,
            metadata={
                "failure_log_path": log_path if failures else None,
                "is_abstention": AbstentionEvaluator.is_abstention(output_text),
            },
        )

        if repair_status:
            result.metadata["repair_status"] = repair_status

        if repaired_prompt:
            result.metadata["repaired_prompt"] = repaired_prompt


        # Store flag for deferred printing
        if failures and hasattr(self, "_failure_log_path"):
            result.metadata["_print_log_path_after"] = True

        return result
    
    def print_predictions(self):
        """
        Print learned failure predictions in a formatted way.
        Handles the display logic internally.
        """
        print("\n" + "=" * 80)
        print("FAILURE PREDICTION (LEARNED)")
        print("=" * 80)
        
        predictions = self.predict_failures()
        
        if not predictions:
            print("No learned failure patterns yet.")
        else:
            for p in predictions:
                print(
                    f"Likely failure: {p.failure_type} | "
                    f"Probability: {p.probability:.2f} | "
                    f"Based on: {p.based_on}"
                )
        
    def get_causal_samples(self):
        """
        Returns causal samples built from collected failures.
        Users should not access internal failure storage directly.
        """
        if not self._causal_failures:
            return []

        builder = CausalDatasetBuilder()
        return builder.build(self._causal_failures)
    
    def visualize_causal_graph(self, show: bool = True, filename: str | None = None):
        """
        Visualizes the learned causal graph if available.
        """
        graph = self.get_causal_graph()
        if graph is None:
             print("\n⚠️ Causal graph not available yet (need ≥2 failures).")
             return None
        viz = CausalGraphVisualizer()

        # Create output directory
        output_dir = os.path.join(os.getcwd(), "causal_outputs")
        os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            provider = self.adapter.__class__.__name__.replace("Adapter", "").lower()
            model = getattr(self.adapter, "model", None) or "default"
            safe_model = model.replace("/", "_").replace(":", "_")
            filename = f"causal_graph_{provider}_{safe_model}_{timestamp}.png"

        output_path = os.path.join(output_dir, filename)
        saved_path = viz.render(graph, filename=output_path, show=show)
        print(f"📈 Causal graph saved to: {saved_path}")

        return saved_path


    def explain_causal_graph(self):
        """
        Explains all causal edges if possible. 
        Requires at least 2 failures to have been observed.
        """
        graph = self.get_causal_graph()
        if graph is None:
            print(
                "\n⚠️ Cannot explain causal relationships yet.\n"
                "Reason: only "
                f"{len(self._causal_failures)} failure(s) observed so far; "
                "at least 2 are required to infer cause–effect patterns."
            )
            return

        samples = self.get_causal_samples()
        viz = CausalGraphVisualizer()
        viz.explain_all_edges(graph, samples)


    def get_causal_graph(self):
        """
        Returns the learned causal graph if available.
        """
        return self._causal_graph
    
    def predict_failures(self) -> List[FailurePrediction]:
        """
        Predict likely failures based purely on learned causal graph.
        No heuristics, no prompt inspection.
        """
        if self._causal_graph is None:
            return []

        predictions: List[FailurePrediction] = []

        for cause, effects in self._causal_graph.edges.items():
            for effect, strength in effects.items():
                predictions.append(
                    FailurePrediction(
                        failure_type=effect,
                        probability=strength,
                        based_on=[cause],
                    )
                )

        return predictions

    @staticmethod
    def _generate_id(text: str, failure_type: FailureType) -> str:
        """Generate unique failure ID"""
        content = f"{text}|{failure_type.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    @staticmethod
    def _determine_severity(failure_type: FailureType, confidence: float) -> Severity:
        """Determine failure severity"""

        # Hallucinations are always serious
        if failure_type == FailureType.HALLUCINATION:
            if confidence > 0.6:
                return Severity.HIGH
            return Severity.MEDIUM

        # Generic confidence-based severity
        if confidence > 0.8:
            return Severity.HIGH
        if confidence > 0.6:
            return Severity.MEDIUM
        return Severity.LOW
    
    def __call__(self, prompt: str, **kwargs) -> DebugResult:
        """
        Callable interface: debugger(prompt)
        """
        return self.debug(prompt, auto_repair=True, **kwargs)



