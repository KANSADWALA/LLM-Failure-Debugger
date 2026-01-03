"""
Evaluation and benchmarking module for debugger performance.

This module provides tools for:
- Computing evaluation metrics (precision, recall, F1)
- Benchmarking debugger performance
- Tracking and analyzing results

Components:
    - EvaluationMetrics: Compute performance metrics
    - Benchmark: Run benchmarks on datasets
"""

from .metrics import EvaluationMetrics
from .benchmark import BenchmarkRunner

__all__ = [
    "EvaluationMetrics",
    "BenchmarkRunner",
]
