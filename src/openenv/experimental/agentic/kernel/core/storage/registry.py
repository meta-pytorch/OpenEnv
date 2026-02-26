# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""In-memory registry for agents."""

import asyncio

from ..config import Agent, AgentState


class AgentRegistry:
    """Thread-safe in-memory registry for agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._lock = asyncio.Lock()

    async def register(self, agent: Agent) -> Agent:
        async with self._lock:
            if agent.id in self._agents:
                raise KeyError(f"Agent already registered: {agent.id}")
            self._agents[agent.id] = agent
            return agent

    async def get(self, agent_id: str) -> Agent | None:
        async with self._lock:
            return self._agents.get(agent_id)

    async def delete(self, agent_id: str) -> None:
        async with self._lock:
            if agent_id not in self._agents:
                raise KeyError(f"Agent not found: {agent_id}")
            del self._agents[agent_id]

    async def update_state(self, agent_id: str, state: AgentState) -> None:
        async with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                raise KeyError(f"Agent not found: {agent_id}")
            agent.state = state

    async def list(
        self,
        team_id: str | None = None,
        state: AgentState | None = None,
        metadata: dict[str, str] | None = None,
    ) -> list[Agent]:
        async with self._lock:
            results = list(self._agents.values())
        if team_id:
            results = [a for a in results if a.team_id == team_id]
        if state:
            results = [a for a in results if a.state == state]
        if metadata:
            results = [
                a
                for a in results
                if all(a.metadata.get(k) == v for k, v in metadata.items())
            ]
        return results
