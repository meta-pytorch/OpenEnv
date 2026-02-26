# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for AgentClient - client that talks to AgentServer via HTTP."""

import asyncio
import json
import socket
from typing import Any, Callable

import pytest
import pytest_asyncio
from agentic.kernel.core.backends.agent_client import AgentClient
from agentic.kernel.core.backends.local.resolver import LocalResolver
from agentic.kernel.core.config import Agent, AgentState, TurnRequest
from agentic.kernel.core.runner.server import AgentServer
from agentic.kernel.core.storage.registry import AgentRegistry

HOST = "127.0.0.1"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


async def _echo_handler(
    messages: list[dict[str, Any]], on_token: Callable[..., Any]
) -> None:
    for msg in messages:
        content = msg.get("content", "")
        for i, word in enumerate(content.split()):
            await on_token(word if i == 0 else f" {word}")
            await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def agent_with_server():
    """Create an AgentServer, register the agent in a registry, yield (client, agent_id)."""
    port = _find_free_port()
    agent_id = "svc-test-agent"

    # Start server
    server = AgentServer({"agent_id": agent_id}, llm_handler=_echo_handler)
    runner = await server.start(port, host=HOST)

    # Register agent in registry
    registry = AgentRegistry()
    agent = Agent(
        id=agent_id,
        name="test",
        team_id="t1",
        agent_type="openclaw",
        image_id="img1",
        http_port=port,
        state=AgentState.RUNNING,
    )
    await registry.register(agent)

    client = AgentClient(LocalResolver(registry))
    yield client, agent_id

    await runner.cleanup()


class TestAgentClient:
    @pytest.mark.asyncio
    async def test_turn_streams_response(
        self, agent_with_server: tuple[AgentClient, str]
    ):
        client, agent_id = agent_with_server

        request = TurnRequest(
            agent_id=agent_id,
            body=json.dumps(
                {"messages": [{"role": "user", "content": "ping pong"}]}
            ).encode(),
        )

        chunks = []
        async for response in client.turn(request):
            chunks.append(response)

        assert len(chunks) >= 2
        assert chunks[-1].done is True

        body = "".join(c.body for c in chunks)
        assert body == "ping pong"

    @pytest.mark.asyncio
    async def test_turn_nonexistent_agent_raises(
        self, agent_with_server: tuple[AgentClient, str]
    ):
        client, _ = agent_with_server

        request = TurnRequest(agent_id="nonexistent", body=b"hi")
        with pytest.raises(KeyError, match="not found"):
            async for _ in client.turn(request):
                pass

    @pytest.mark.asyncio
    async def test_get_history(self, agent_with_server: tuple[AgentClient, str]):
        client, agent_id = agent_with_server

        # Do a turn first to create history
        request = TurnRequest(
            agent_id=agent_id,
            body=json.dumps(
                {"messages": [{"role": "user", "content": "hello"}]}
            ).encode(),
        )
        async for _ in client.turn(request):
            pass

        # Get history
        history = await client.get_history(agent_id)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_get_history_nonexistent_raises(
        self, agent_with_server: tuple[AgentClient, str]
    ):
        client, _ = agent_with_server
        with pytest.raises(KeyError, match="not found"):
            await client.get_history("nonexistent")
