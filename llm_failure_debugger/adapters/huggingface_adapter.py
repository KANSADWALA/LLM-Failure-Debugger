from typing import Optional

from .custom_adapter import CustomAdapter
from ..type_definitions import ModelCapabilities, ModelInternals


class HuggingFaceAdapter(CustomAdapter):
    """
    Adapter for Hugging Face Transformers models (local).

    Example models:
    - gpt2
    - mistralai/Mistral-7B-Instruct
    - meta-llama/Llama-2-7b-chat-hf
    """

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        super().__init__()

    def _declare_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_logits=False,
            supports_embeddings=False,
            supports_nli=False,
            supports_rag=False,
            supports_tools=False,
            supports_multi_output=False,
        )

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            )

        self._pipeline = pipeline(
            "text-generation",
            model=self.model,
            device=self.device,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        self._load_pipeline()

        outputs = self._pipeline(
            prompt,
            max_new_tokens=kwargs.get("max_tokens", self.max_new_tokens),
            do_sample=True,
        )

        # HF pipelines return list[dict]
        return outputs[0]["generated_text"]

    def extract_internals(self, prompt, output, **kwargs) -> ModelInternals:
        # Hugging Face pipeline does not expose internals by default
        return ModelInternals()
