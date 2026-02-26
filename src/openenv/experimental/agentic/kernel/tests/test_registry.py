# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for AgentRegistry."""

import asyncio

import pytest
from agentic.kernel.core.config import Agent, AgentState
from agentic.kernel.core.storage.registry import AgentRegistry


@pytest.fixture
def agent_registry() -> AgentRegistry:
    return AgentRegistry()


def _make_agent(id: str = "a1", **kwargs) -> Agent:
    defaults = {
        "id": id,
        "name": "test-agent",
        "team_id": "team1",
        "agent_type": "openclaw",
        "image_id": "img1",
    }
    defaults.update(kwargs)
    return Agent(**defaults)


class TestAgentRegistry:
    @pytest.mark.asyncio
    async def test_register_and_get(self, agent_registry: AgentRegistry):
        agent = _make_agent()
        await agent_registry.register(agent)
        retrieved = await agent_registry.get("a1")
        assert retrieved is not None
        assert retrieved.name == "test-agent"

    @pytest.mark.asyncio
    async def test_register_duplicate_raises(self, agent_registry: AgentRegistry):
        await agent_registry.register(_make_agent())
        with pytest.raises(KeyError, match="already registered"):
            await agent_registry.register(_make_agent())

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, agent_registry: AgentRegistry):
        assert await agent_registry.get("nope") is None

    @pytest.mark.asyncio
    async def test_delete(self, agent_registry: AgentRegistry):
        await agent_registry.register(_make_agent())
        await agent_registry.delete("a1")
        assert await agent_registry.get("a1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self, agent_registry: AgentRegistry):
        with pytest.raises(KeyError, match="not found"):
            await agent_registry.delete("nope")

    @pytest.mark.asyncio
    async def test_update_state(self, agent_registry: AgentRegistry):
        await agent_registry.register(_make_agent())
        await agent_registry.update_state("a1", AgentState.RUNNING)
        agent = await agent_registry.get("a1")
        assert agent.state == AgentState.RUNNING

    @pytest.mark.asyncio
    async def test_update_state_nonexistent_raises(self, agent_registry: AgentRegistry):
        with pytest.raises(KeyError):
            await agent_registry.update_state("nope", AgentState.RUNNING)

    @pytest.mark.asyncio
    async def test_list_with_filters(self, agent_registry: AgentRegistry):
        await agent_registry.register(
            _make_agent("a1", team_id="t1", metadata={"role": "worker"})
        )
        await agent_registry.register(
            _make_agent("a2", team_id="t1", metadata={"role": "reviewer"})
        )
        await agent_registry.register(
            _make_agent("a3", team_id="t2", metadata={"role": "worker"})
        )

        assert len(await agent_registry.list()) == 3
        assert len(await agent_registry.list(metadata={"role": "worker"})) == 2
        assert len(await agent_registry.list(team_id="t1")) == 2
        assert (
            len(await agent_registry.list(metadata={"role": "worker"}, team_id="t1"))
            == 1
        )

    @pytest.mark.asyncio
    async def test_list_by_state(self, agent_registry: AgentRegistry):
        a1 = _make_agent("a1")
        a2 = _make_agent("a2")
        await agent_registry.register(a1)
        await agent_registry.register(a2)
        await agent_registry.update_state("a1", AgentState.RUNNING)

        running = await agent_registry.list(state=AgentState.RUNNING)
        assert len(running) == 1
        assert running[0].id == "a1"

    @pytest.mark.asyncio
    async def test_concurrent_registers(self, agent_registry: AgentRegistry):
        agents = [_make_agent(f"a{i}") for i in range(20)]
        await asyncio.gather(*[agent_registry.register(a) for a in agents])
        all_agents = await agent_registry.list()
        assert len(all_agents) == 20
