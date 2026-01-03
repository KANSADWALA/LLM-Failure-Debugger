import json
import requests

from .custom_adapter import CustomAdapter
from ..type_definitions import ModelCapabilities, ModelInternals


class OllamaAdapter(CustomAdapter):
    """
    Adapter for Ollama-served local models (LLaMA, Mistral, Phi, etc.).

    Assumes Ollama is running locally on http://localhost:11434
    """

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        super().__init__()

    # 🔴 REQUIRED
    def _declare_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_logits=False,
            supports_embeddings=False,
            supports_nli=False,
            supports_rag=False,
            supports_tools=False,
            supports_multi_output=False,
        )

    # 🔴 REQUIRED
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 300),
                },
            },
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            if "response" in payload:
                chunks.append(payload["response"])

            if payload.get("done"):
                break

        return "".join(chunks)

    # 🔴 REQUIRED
    def extract_internals(self, prompt: str, output: str, **kwargs) -> ModelInternals:
        # Ollama does not expose internals yet
        return ModelInternals()
