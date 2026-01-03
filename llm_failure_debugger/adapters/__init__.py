"""
Adapter implementations for different LLM providers.

This module provides adapter classes that handle model-specific integration
and extract internal signals for failure analysis.

Available Adapters:
    - LLMAdapter: Base abstract class for all adapters
    - OpenAIAdapter: For OpenAI models (GPT-4, GPT-3.5)
    - AnthropicAdapter: For Anthropic Claude models
    - CustomAdapter: For implementing custom model integrations
"""

from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .ollama_adapter import OllamaAdapter

__all__ = [
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
]

