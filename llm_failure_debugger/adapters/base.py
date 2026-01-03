
"""
base.py defines the RULES of an adapter, says: “Any LLM adapter MUST be able to generate text and (optionally) expose internals.
"""

from abc import ABC, abstractmethod
from typing import Optional
from ..type_definitions import ModelCapabilities, ModelInternals


class LLMAdapter(ABC):
    """
    Base class for LLM adapters.
    
    Adapters handle model-specific integration and extract internal signals.
    """
    
    def __init__(self):
        self._capabilities = self._declare_capabilities()
    
    @abstractmethod
    def _declare_capabilities(self) -> ModelCapabilities:
        """Declare what signals this adapter can extract"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate output from the model"""
        pass
    
    @abstractmethod
    def extract_internals(
        self, 
        prompt: str, 
        output: str, 
        **kwargs
    ) -> ModelInternals:
        """
        Extract internal signals from the model.
        
        This is where adapter-specific logic goes to collect:
        - Token probabilities/entropy
        - RAG grounding info
        - Tool traces
        - Embeddings
        - etc.
        """
        pass
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities"""
        return self._capabilities
    
    def analyze(self, prompt: str, output: Optional[str] = None, **kwargs) -> tuple[str, ModelInternals]:
        """
        Convenience method: generate + extract in one call.
        
        If output is provided, skip generation.
        """
        if output is None:
            output = self.generate(prompt, **kwargs)
        
        internals = self.extract_internals(prompt, output, **kwargs)

        self._validate_capabilities(internals)

        return output, internals
    
    def _validate_capabilities(self, internals: ModelInternals):
        """
        Warn if adapter capabilities do not match extracted internals.
        """
        caps = self._capabilities

        warnings = []

        if caps.supports_logits and not internals.token_probs:
            warnings.append("supports_logits=True but token_probs is empty")

        if caps.supports_rag and not internals.rag:
            warnings.append("supports_rag=True but rag info is missing")

        if caps.supports_tools and not internals.tool_trace:
            warnings.append("supports_tools=True but tool_trace is missing")

        if caps.supports_multi_output and not internals.multi_outputs:
            warnings.append("supports_multi_output=True but multi_outputs is missing")

        if warnings:
            import warnings as _warnings
            for w in warnings:
                _warnings.warn(
                    f"[Adapter Validation] {self.__class__.__name__}: {w}",
                    RuntimeWarning,
                )

