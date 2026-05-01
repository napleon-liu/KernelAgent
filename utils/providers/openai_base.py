# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base provider for OpenAI-compatible APIs."""

from typing import Any
import logging
import os

import httpx

from .base import BaseProvider, LLMResponse
from .env_config import configure_proxy_environment


class OpenAICompatibleProvider(BaseProvider):
    """Base provider for OpenAI-compatible APIs."""

    def __init__(self, api_key_env: str, base_url: str | None = None):
        self.api_key_env = api_key_env
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._original_proxy_env = None
        super().__init__()

    def _initialize_client(self) -> None:
        """Initialize HTTP client for OpenAI-compatible API."""
        api_key = self._get_api_key(self.api_key_env)
        if not api_key:
            return

        # Configure proxy using centralized utility function
        self._original_proxy_env = configure_proxy_environment()

        self.client = httpx.Client(
            base_url=(self.base_url or "https://api.openai.com/v1").rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(600.0),
        )

    def get_response(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Get single response."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, **kwargs)
        response = self.client.post("chat/completions", json=api_params)
        response.raise_for_status()
        payload = response.json()
        logging.getLogger(__name__).info(
            "OpenAI-compatible chat response (single): %s",
            payload,
        )

        return self._build_llm_response(model_name, payload)

    def get_multiple_responses(
        self, model_name: str, messages: list[dict[str, str]], n: int = 1, **kwargs
    ) -> list[LLMResponse]:
        """Get multiple responses using n parameter."""
        if not self.is_available():
            raise RuntimeError(f"{self.name} client not available")

        api_params = self._build_api_params(model_name, messages, n=n, **kwargs)
        response = self.client.post("chat/completions", json=api_params)
        response.raise_for_status()
        payload = response.json()
        logging.getLogger(__name__).info(
            "OpenAI-compatible chat response (multi): %s",
            payload,
        )

        usage = payload.get("usage")
        response_id = payload.get("id")
        return [
            LLMResponse(
                content=choice["message"]["content"],
                model=model_name,
                provider=self.name,
                usage=usage,
                response_id=response_id,
            )
            for choice in payload.get("choices", [])
        ]

    def _build_api_params(
        self, model_name: str, messages: list[dict[str, str]], **kwargs
    ) -> dict[str, Any]:
        """Build API parameters for OpenAI-compatible call."""
        params = {
            "model": model_name,
            "messages": messages,
        }

        # GPT-5 and o-series models pin their own sampling behaviour
        if not (model_name.startswith("gpt-5") or model_name.startswith("o")):
            params["temperature"] = kwargs.get("temperature", 0.7)

        # Use max_completion_tokens for newer models like GPT-5, fallback to max_tokens
        max_tokens_value = min(
            kwargs.get("max_tokens", 8192), self.get_max_tokens_limit(model_name)
        )
        if model_name.startswith("gpt-5") or model_name.startswith("o"):
            params["max_completion_tokens"] = max_tokens_value
        else:
            params["max_tokens"] = max_tokens_value

        # Add n parameter if specified
        if "n" in kwargs:
            params["n"] = kwargs["n"]

        # Keep reasoning effort conservative on reasoning-capable models to
        # reduce gateway timeouts on OpenAI-compatible backends.
        if kwargs.get("high_reasoning_effort") and model_name.startswith(
            ("gpt-5", "o3", "o1")
        ):
            params["reasoning_effort"] = "medium"

        return params

    def _build_llm_response(self, model_name: str, payload: dict[str, Any]) -> LLMResponse:
        """Convert OpenAI-compatible JSON payload into an LLMResponse."""
        choice = payload["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=model_name,
            provider=self.name,
            usage=payload.get("usage"),
            response_id=payload.get("id"),
        )

    def is_available(self) -> bool:
        """Check if provider is available."""
        return self.client is not None

    def supports_multiple_completions(self) -> bool:
        """OpenAI-compatible APIs support native multiple completions."""
        return True
