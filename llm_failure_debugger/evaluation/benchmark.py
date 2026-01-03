"""
Benchmark runner for testing debugger on datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from ..core.debugger import Debugger
from ..type_definitions import ModelInternals
from .metrics import EvaluationMetrics


@dataclass
class BenchmarkSample:
    """Single benchmark sample"""
    id: str
    input: str
    output: str
    internals: ModelInternals
    ground_truth: Dict[str, Any]


class BenchmarkRunner:
    """
    Run debugger on benchmark datasets.
    
    Example:
        >>> runner = BenchmarkRunner(debugger)
        >>> results = runner.run_from_file("benchmark.json")
        >>> print(f"F1: {results['metrics']['f1']}")
    """
    
    def __init__(self, debugger: Debugger):
        self.debugger = debugger
        self.metrics = EvaluationMetrics()
    
    def run_from_file(self, filepath: str) -> Dict[str, Any]:
        """
        Run benchmark from JSON file.
        
        File format:
        [
            {
                "id": "sample_1",
                "input": "What is 2+2?",
                "output": "5",
                "internals": {...},
                "ground_truth": {
                    "failure_type": "hallucination",
                    "root_causes": ["ungrounded_generation"]
                }
            },
            ...
        ]
        """
        with open(filepath) as f:
            data = json.load(f)
        
        samples = [self._parse_sample(item) for item in data]
        return self.run(samples)
    
    def run(self, samples: List[BenchmarkSample]) -> Dict[str, Any]:
        """Run benchmark on samples"""
        all_predictions = []
        all_ground_truth = []
        
        for sample in samples:
            result = self.debugger.analyze(
                sample.input,
                sample.output,
                sample.internals
            )
            
            all_predictions.extend(result.failures)
            all_ground_truth.append(sample.ground_truth)
        
        # Compute metrics
        detection = self.metrics.detection_metrics(
            all_predictions, all_ground_truth
        )
        
        root_cause_f1 = self.metrics.root_cause_f1(
            all_predictions, all_ground_truth
        )
        
        return {
            "metrics": {
                **detection,
                "root_cause_f1": root_cause_f1,
            },
            "num_samples": len(samples),
            "num_predictions": len(all_predictions),
        }
    
    def _parse_sample(self, data: Dict[str, Any]) -> BenchmarkSample:
        """Parse sample from dict"""
        internals_dict = data.get("internals", {})
        internals = ModelInternals(
            token_probs=internals_dict.get("token_probs"),
            rag=internals_dict.get("rag"),
            nli_pairs=internals_dict.get("nli_pairs"),
            tool_trace=internals_dict.get("tool_trace"),
            tool_results=internals_dict.get("tool_results"),
            multi_outputs=internals_dict.get("multi_outputs"),
            output_embeddings=internals_dict.get("output_embeddings"),
        )
        
        return BenchmarkSample(
            id=data["id"],
            input=data["input"],
            output=data["output"],
            internals=internals,
            ground_truth=data["ground_truth"],
        )