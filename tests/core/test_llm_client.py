# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for LLMClient abstraction and OpenAIClient implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openenv.core.llm_client import LLMClient, OpenAIClient


class TestLLMClientABC:
    """Test the abstract base class."""

    def test_cannot_instantiate_directly(self):
        """LLMClient is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            LLMClient("http://localhost", 8000)

    def test_concrete_subclass(self):
        """A concrete subclass can be instantiated."""

        class StubClient(LLMClient):
            async def complete(self, prompt: str, **kwargs) -> str:
                return "stub"

        client = StubClient("http://localhost", 8000)
        assert client.endpoint == "http://localhost"
        assert client.port == 8000

    def test_base_url_property(self):
        """base_url combines endpoint and port."""

        class StubClient(LLMClient):
            async def complete(self, prompt: str, **kwargs) -> str:
                return "stub"

        client = StubClient("http://localhost", 8000)
        assert client.base_url == "http://localhost:8000"

    def test_base_url_custom_endpoint(self):
        """base_url works with custom endpoints."""

        class StubClient(LLMClient):
            async def complete(self, prompt: str, **kwargs) -> str:
                return "stub"

        client = StubClient("https://api.example.com", 443)
        assert client.base_url == "https://api.example.com:443"


class TestOpenAIClientConstruction:
    """Test OpenAIClient initialization."""

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_basic_construction(self, mock_openai_cls):
        """OpenAIClient stores params and creates AsyncOpenAI."""
        client = OpenAIClient("http://localhost", 8000, model="gpt-4")

        assert client.endpoint == "http://localhost"
        assert client.port == 8000
        assert client.model == "gpt-4"
        assert client.temperature == 0.0
        assert client.max_tokens == 256
        assert client.system_prompt is None

        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
        )

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_custom_api_key(self, mock_openai_cls):
        """API key is passed through to AsyncOpenAI."""
        OpenAIClient("http://localhost", 8000, model="gpt-4", api_key="sk-test-123")

        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="sk-test-123",
        )

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_default_api_key_when_none(self, mock_openai_cls):
        """api_key=None defaults to 'not-needed'."""
        OpenAIClient("http://localhost", 8000, model="gpt-4", api_key=None)

        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
        )

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_system_prompt_stored(self, mock_openai_cls):
        """System prompt is stored for use in complete()."""
        client = OpenAIClient(
            "http://localhost",
            8000,
            model="gpt-4",
            system_prompt="You are a judge.",
        )
        assert client.system_prompt == "You are a judge."

    @patch("openenv.core.llm_client.AsyncOpenAI")
    def test_custom_temperature_and_max_tokens(self, mock_openai_cls):
        """Custom temperature and max_tokens are stored."""
        client = OpenAIClient(
            "http://localhost",
            8000,
            model="gpt-4",
            temperature=0.7,
            max_tokens=512,
        )
        assert client.temperature == 0.7
        assert client.max_tokens == 512


class TestOpenAIClientComplete:
    """Test the complete() method."""

    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(self):
        """complete() sends user message only when no system prompt."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "42"
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient("http://localhost", 8000, model="gpt-4")
            result = await client.complete("What is 2+2?")

        assert result == "42"
        mock_openai.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.0,
            max_tokens=256,
        )

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self):
        """complete() includes system message when system_prompt is set."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "0.8"
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient(
                "http://localhost",
                8000,
                model="gpt-4",
                system_prompt="You are a judge.",
            )
            result = await client.complete("Rate this code.")

        assert result == "0.8"
        mock_openai.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": "Rate this code."},
            ],
            temperature=0.0,
            max_tokens=256,
        )

    @pytest.mark.asyncio
    async def test_complete_kwargs_override(self):
        """Keyword arguments override default temperature and max_tokens."""
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openenv.core.llm_client.AsyncOpenAI", return_value=mock_openai):
            client = OpenAIClient("http://localhost", 8000, model="gpt-4")
            await client.complete("hi", temperature=0.9, max_tokens=100)

        mock_openai.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.9,
            max_tokens=100,
        )
