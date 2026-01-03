"""
Template for custom LLM adapters.
"""

from .base import LLMAdapter
from ..type_definitions import ModelCapabilities, ModelInternals


class CustomAdapter(LLMAdapter):
    """
    Template adapter for custom models.
    
    Use this as a starting point for integrating your own LLM.
    
    Example:
        >>> class MyModelAdapter(CustomAdapter):
        ...     def generate(self, prompt: str, **kwargs) -> str:
        ...         return my_model.generate(prompt)
        ...     
        ...     def extract_internals(self, prompt, output, **kwargs):
        ...         return ModelInternals(
        ...             token_probs=my_model.get_token_probs()
        ...         )
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
        super().__init__()

    def _declare_capabilities(self) -> ModelCapabilities:
        # Default: no special capabilities unless explicitly declared
        return ModelCapabilities()
  
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Override this to implement your generation logic.
        
        Example:
            return your_model.generate(prompt, **kwargs)
        """
        raise NotImplementedError("Implement generation logic for your model")
    
    def extract_internals(self, prompt: str, output: str, **kwargs) -> ModelInternals:
        """
        Override this to extract model-specific signals.
        
        Example:
            return ModelInternals(
                token_probs=your_model.get_logprobs(),
                rag=your_model.get_rag_info(),
            )
        """
        return ModelInternals()

