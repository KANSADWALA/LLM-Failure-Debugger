"""
Anthropic Claude adapter implementation.
"""

from .base import LLMAdapter
from ..type_definitions import ModelCapabilities, ModelInternals

class AnthropicAdapter(LLMAdapter):
    """
    Adapter for Anthropic Claude models.
    
    Example:
        >>> adapter = AnthropicAdapter(api_key="sk-ant-...")
        >>> output, internals = adapter.analyze("Explain quantum physics")
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229", **default_kwargs):
        self.api_key = api_key
        self.model = model
        self.default_kwargs = default_kwargs
        super().__init__()
    
    def _declare_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_logits=False,
            supports_embeddings=False,
            supports_nli=False,
            supports_tools=False,
            supports_rag=False,
            supports_multi_output=False,
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using Anthropic API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            params = {**self.default_kwargs, **kwargs}
            
            message = client.messages.create(
                model=self.model,
                max_tokens=params.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
        
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
    
    def extract_internals(self, prompt: str, output: str, **kwargs) -> ModelInternals:
        """Extract internal signals (limited for Claude)"""
        # Claude doesn't expose much internal state yet
        # Can be extended when more signals become available
        return ModelInternals()