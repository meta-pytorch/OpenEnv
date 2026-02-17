# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LLM client abstraction for calling LLM endpoints.

Provides a generic RPC abstraction: point it at an endpoint/port, tell it the
protocol, and it works. OpenAI-compatible API is the first implementation,
covering OpenAI, vLLM, TGI, Ollama, HuggingFace Inference API, etc.

Usage:
    client = OpenAIClient("http://localhost", 8000, model="meta-llama/...")
    response = await client.complete("What is 2+2?")
"""

from abc import ABC, abstractmethod

from openai import AsyncOpenAI


class LLMClient(ABC):
    """Abstract base for LLM endpoint clients.

    Subclass and implement ``complete()`` for your protocol.

    Args:
        endpoint: The base URL of the LLM service (e.g. "http://localhost").
        port: The port the service listens on.
    """

    def __init__(self, endpoint: str, port: int):
        self.endpoint = endpoint
        self.port = port

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Send a prompt, return the text response.

        Args:
            prompt: The user prompt to send.
            **kwargs: Override default parameters (temperature, max_tokens, etc.).

        Returns:
            The model's text response.
        """
        ...

    @property
    def base_url(self) -> str:
        """Construct base URL from endpoint and port."""
        return f"{self.endpoint}:{self.port}"


class OpenAIClient(LLMClient):
    """Client for OpenAI-compatible APIs.

    Works with: OpenAI, vLLM, TGI, Ollama, HuggingFace Inference API,
    or any endpoint that speaks the OpenAI chat completions format.

    Args:
        endpoint: The base URL (e.g. "http://localhost").
        port: The port number.
        model: Model name to pass to the API.
        api_key: API key. Defaults to "not-needed" for local endpoints.
        system_prompt: Optional system message prepended to every request.
        temperature: Default sampling temperature.
        max_tokens: Default max tokens in the response.
    """

    def __init__(
        self,
        endpoint: str,
        port: int,
        model: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        super().__init__(endpoint, port)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AsyncOpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key if api_key is not None else "not-needed",
        )

    async def complete(self, prompt: str, **kwargs) -> str:
        """Send a chat completion request.

        Args:
            prompt: The user message.
            **kwargs: Overrides for temperature, max_tokens.

        Returns:
            The assistant's response text.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content
