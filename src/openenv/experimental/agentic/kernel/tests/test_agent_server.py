# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for AgentServer - real HTTP server with mock LLM handler."""

import asyncio
import json
import socket
from typing import Any, Callable

import aiohttp
import pytest
import pytest_asyncio
from agentic.kernel.core.runner.server import AgentServer

HOST = "127.0.0.1"


def _find_free_port() -> int:
    """Find a free port by binding to port 0 on IPv4."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


async def _echo_handler(
    messages: list[dict[str, Any]], on_token: Callable[..., Any]
) -> None:
    """Mock LLM handler that echoes back user messages as tokens."""
    for msg in messages:
        content = msg.get("content", "")
        words = content.split()
        for i, word in enumerate(words):
            token = word if i == 0 else f" {word}"
            await on_token(token)
            await asyncio.sleep(0.01)


async def _error_handler(
    messages: list[dict[str, Any]], on_token: Callable[..., Any]
) -> None:
    """Mock LLM handler that raises an error."""
    raise RuntimeError("LLM call failed")


@pytest_asyncio.fixture
async def echo_server():
    """Start an AgentServer with echo handler, yield (port, server), then cleanup."""
    port = _find_free_port()
    config = {"agent_id": "test-agent"}
    server = AgentServer(config, llm_handler=_echo_handler)
    runner = await server.start(port, host=HOST)
    yield port, server
    await runner.cleanup()


@pytest_asyncio.fixture
async def error_server():
    """Start an AgentServer with error handler."""
    port = _find_free_port()
    config = {"agent_id": "test-agent"}
    server = AgentServer(config, llm_handler=_error_handler)
    runner = await server.start(port, host=HOST)
    yield port, server
    await runner.cleanup()


def _url(port: int, path: str) -> str:
    return f"http://{HOST}:{port}{path}"


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, echo_server: tuple[int, AgentServer]):
        port, _ = echo_server
        async with aiohttp.ClientSession() as session:
            async with session.get(_url(port, "/health")) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["agent_id"] == "test-agent"


class TestTurnEndpoint:
    @pytest.mark.asyncio
    async def test_turn_streams_response(self, echo_server: tuple[int, AgentServer]):
        port, _ = echo_server
        payload = {
            "agent_id": "test-agent",
            "body": json.dumps(
                {"messages": [{"role": "user", "content": "hello world"}]}
            ),
        }

        chunks = []
        async with aiohttp.ClientSession() as session:
            async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                assert resp.status == 200
                assert resp.content_type == "text/event-stream"

                async for line in resp.content:
                    decoded = line.decode().strip()
                    if decoded.startswith("data: "):
                        chunks.append(json.loads(decoded[6:]))

        # Should have token chunks + final done chunk
        assert len(chunks) >= 2
        assert chunks[-1]["done"] is True

        # Reconstruct the streamed response
        body = "".join(c["body"] for c in chunks)
        assert body == "hello world"

    @pytest.mark.asyncio
    async def test_turn_wrong_agent_id_returns_400(
        self, echo_server: tuple[int, AgentServer]
    ):
        port, _ = echo_server
        payload = {"agent_id": "wrong-agent", "body": "hi"}
        async with aiohttp.ClientSession() as session:
            async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                assert resp.status == 400
                data = await resp.json()
                assert "error" in data

    @pytest.mark.asyncio
    async def test_turn_plain_string_body(self, echo_server: tuple[int, AgentServer]):
        """Body as plain string (not JSON) should still work."""
        port, _ = echo_server
        payload = {
            "agent_id": "test-agent",
            "body": "just a string",
        }

        chunks = []
        async with aiohttp.ClientSession() as session:
            async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                async for line in resp.content:
                    decoded = line.decode().strip()
                    if decoded.startswith("data: "):
                        chunks.append(json.loads(decoded[6:]))

        body = "".join(c["body"] for c in chunks)
        assert body == "just a string"

    @pytest.mark.asyncio
    async def test_turn_error_propagated_in_stream(
        self, error_server: tuple[int, AgentServer]
    ):
        port, _ = error_server
        payload = {
            "agent_id": "test-agent",
            "body": json.dumps(
                {"messages": [{"role": "user", "content": "trigger error"}]}
            ),
        }

        chunks = []
        async with aiohttp.ClientSession() as session:
            async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                async for line in resp.content:
                    decoded = line.decode().strip()
                    if decoded.startswith("data: "):
                        chunks.append(json.loads(decoded[6:]))

        assert any(c.get("error") for c in chunks)
        assert chunks[-1]["done"] is True


class TestHistoryEndpoint:
    @pytest.mark.asyncio
    async def test_history_records_turns(self, echo_server: tuple[int, AgentServer]):
        port, _ = echo_server

        # Do a turn first
        payload = {
            "agent_id": "test-agent",
            "body": json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                async for _ in resp.content:
                    pass

            async with session.post(_url(port, "/v1/history"), json={}) as resp:
                data = await resp.json()
                entries = data["entries"]
                assert len(entries) == 2  # user + assistant
                assert entries[0]["role"] == "user"
                assert entries[1]["role"] == "assistant"
                assert entries[1]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_history_last_n(self, echo_server: tuple[int, AgentServer]):
        port, server = echo_server

        for i in range(6):
            server.history.append({"role": "user", "content": f"msg-{i}"})

        async with aiohttp.ClientSession() as session:
            async with session.post(
                _url(port, "/v1/history"), json={"last_n": 2}
            ) as resp:
                data = await resp.json()
                entries = data["entries"]
                assert len(entries) == 2
                assert entries[0]["content"] == "msg-4"
                assert entries[1]["content"] == "msg-5"


class TestNonceValidation:
    @pytest.mark.asyncio
    async def test_turn_with_valid_nonce_succeeds(self):
        port = _find_free_port()
        config = {"agent_id": "nonce-agent", "nonce": "secret-token"}
        server = AgentServer(config, llm_handler=_echo_handler)
        runner = await server.start(port, host=HOST)
        try:
            payload = {
                "agent_id": "nonce-agent",
                "nonce": "secret-token",
                "body": json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                    assert resp.status == 200
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_turn_with_wrong_nonce_returns_403(self):
        port = _find_free_port()
        config = {"agent_id": "nonce-agent", "nonce": "secret-token"}
        server = AgentServer(config, llm_handler=_echo_handler)
        runner = await server.start(port, host=HOST)
        try:
            payload = {
                "agent_id": "nonce-agent",
                "nonce": "wrong-token",
                "body": "hi",
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                    assert resp.status == 403
                    data = await resp.json()
                    assert "nonce" in data["error"].lower()
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_turn_with_missing_nonce_returns_403(self):
        port = _find_free_port()
        config = {"agent_id": "nonce-agent", "nonce": "secret-token"}
        server = AgentServer(config, llm_handler=_echo_handler)
        runner = await server.start(port, host=HOST)
        try:
            payload = {"agent_id": "nonce-agent", "body": "hi"}
            async with aiohttp.ClientSession() as session:
                async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                    assert resp.status == 403
        finally:
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_turn_without_nonce_in_config_skips_validation(self):
        """When config has no nonce, any request is accepted (backwards compat)."""
        port = _find_free_port()
        config = {"agent_id": "no-nonce-agent"}
        server = AgentServer(config, llm_handler=_echo_handler)
        runner = await server.start(port, host=HOST)
        try:
            payload = {"agent_id": "no-nonce-agent", "body": "hi"}
            async with aiohttp.ClientSession() as session:
                async with session.post(_url(port, "/v1/turn"), json=payload) as resp:
                    assert resp.status == 200
        finally:
            await runner.cleanup()


class TestNoHandler:
    @pytest.mark.asyncio
    async def test_turn_without_handler_returns_503(self):
        port = _find_free_port()
        server = AgentServer({"agent_id": "no-llm"}, llm_handler=None)
        runner = await server.start(port, host=HOST)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    _url(port, "/v1/turn"),
                    json={"agent_id": "no-llm", "body": "hi"},
                ) as resp:
                    assert resp.status == 503
        finally:
            await runner.cleanup()
