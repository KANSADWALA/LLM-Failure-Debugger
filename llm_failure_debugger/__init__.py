"""
LLM Failure Debugger

A simple, production-ready library for diagnosing and fixing LLM failures.

Typical usage:

    from llm_failure_debugger import Debugger

    debugger = Debugger.from_openai(api_key="sk-...")
    result = debugger("What is the capital of France?")

    if result.has_failures:
        print(result.recommendations)
        print(result.repaired_prompt)
"""

# Core debugger
from .core.debugger import Debugger

# Adapter base (for custom / any LLM)
from .adapters.custom_adapter import CustomAdapter

# Benchmarking
from .evaluation.benchmark import BenchmarkRunner

# Utility functions for prompt repair
from .utils.repair import RepairEvaluator

# Type definitions (for advanced / custom usage)
from .type_definitions import (
    ModelCapabilities,
    ModelInternals,
)

__version__ = "0.1.0"

__all__ = [
    "Debugger",
    "CustomAdapter",
    "BenchmarkRunner",
    "RepairEvaluator",   
    "ModelCapabilities",
    "ModelInternals",
]
