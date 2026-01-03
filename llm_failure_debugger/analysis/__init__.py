"""
Analysis module for failure diagnosis and remediation.

This module provides tools for:
- Root cause analysis: Maps signals to underlying failure causes
- Recommendation generation: Suggests mitigations for identified issues

Components:
    - RootCauseAnalyzer: Analyzes failure signals to identify root causes
    - RecommendationEngine: Generates actionable recommendations
"""

from .root_cause import RootCauseAnalyzer
from .recommendations import RecommendationEngine

__all__ = [
    "RootCauseAnalyzer",
    "RecommendationEngine",
]
