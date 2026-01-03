"""
OpenAI adapter implementation.
"""

import math
from typing import Optional

from .base import LLMAdapter
from ..type_definitions import ModelCapabilities, ModelInternals


class OpenAIAdapter(LLMAdapter):
    """
    Adapter for OpenAI ChatCompletion-based models.

    - Calls OpenAI API exactly ONCE per prompt
    - Caches response for internal signal extraction
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        **default_kwargs,
    ):
        self.api_key = api_key
        self.model = model
        self.default_kwargs = default_kwargs

        # cache for last response (IMPORTANT)
        self._last_response = None

        super().__init__()

    # --------------------------------------------------
    # Capabilities
    # --------------------------------------------------
    def _declare_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_logits=True,
            supports_embeddings=False,
            supports_nli=False,
            supports_rag=False,
            supports_tools=True,
            supports_multi_output=False,
        )

    # --------------------------------------------------
    # Generation (SINGLE API CALL)
    # --------------------------------------------------
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import openai

            openai.api_key = self.api_key

            params = {
                **self.default_kwargs,
                **kwargs,
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "logprobs": True,
                "top_logprobs": 5,
            }

            response = openai.ChatCompletion.create(**params)

            # cache response for internals extraction
            self._last_response = response

            return response.choices[0].message.content

        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    # --------------------------------------------------
    # Internals extraction (NO API CALL)
    # --------------------------------------------------
    def extract_internals(
        self,
        prompt: str,
        output: str,
        **kwargs,
    ) -> ModelInternals:
        response = self._last_response
        if response is None:
            # generation was skipped or failed
            return ModelInternals()

        # ---------------------------
        # Token probabilities / entropy (proxy)
        # ---------------------------
        token_probs = []
        choice = response.choices[0]

        logprobs = getattr(choice, "logprobs", None)
        if logprobs and getattr(logprobs, "content", None):
            for token_data in logprobs.content:
                logprob = token_data.logprob
                prob = math.exp(logprob)

                # NOTE: proxy entropy, not true Shannon entropy
                entropy = -logprob * prob

                token_probs.append(
                    {
                        "token": token_data.token,
                        "logprob": logprob,
                        "prob": prob,
                        "entropy": entropy,
                    }
                )

        # ---------------------------
        # Tool trace (if any)
        # ---------------------------
        tool_trace = None
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            tool_trace = {
                "invoked_tools": [
                    tc.function.name for tc in tool_calls
                ]
            }

        return ModelInternals(
            token_probs=token_probs or None,
            tool_trace=tool_trace,
        )
