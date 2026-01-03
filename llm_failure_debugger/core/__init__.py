"""
Core debugging engine for LLM failure analysis.

This module contains the main components of the debugging pipeline:
- Signal extraction and inference
- Failure detection and analysis
- Integration with adapters

Components:
    - Debugger: Main public API for failure analysis
    - FailureInferenceEngine: Detects failures from signals
    - Extractors: Extract various failure signals
    - CausalModel: Models causal relationships between signals
"""

from .debugger import Debugger
from .inference import FailureInferenceEngine
from .signals import EXTRACTORS
from .causal_model import CausalModel

__all__ = [
    "Debugger",
    "FailureInferenceEngine",
    "EXTRACTORS",
    "CausalModel",
]
