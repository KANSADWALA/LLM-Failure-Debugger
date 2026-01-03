import os
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .ollama_adapter import OllamaAdapter
from .huggingface_adapter import HuggingFaceAdapter


def create_adapter(provider: str, model: str | None = None, **kwargs):
    provider = provider.lower()

    if provider == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIAdapter(
            api_key=api_key,
            model=model or "gpt-4",
        )

    if provider == "anthropic":
        api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return AnthropicAdapter(
            api_key=api_key,
            model=model or "claude-3-opus-20240229",
        )

    if provider == "ollama":
        return OllamaAdapter(
            model=model or "llama2",
        )
    
    if provider == "huggingface":
        if not model:
            raise ValueError("model name required for Hugging Face adapter")
        return HuggingFaceAdapter(
            model=model,
            **kwargs,
        )


    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Supported: openai, anthropic, ollama"
    )
