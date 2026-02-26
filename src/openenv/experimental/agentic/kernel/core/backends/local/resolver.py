# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""LocalResolver - resolves agents to localhost:{http_port}."""

from ...storage.registry import AgentRegistry


class LocalResolver:
    """Resolves agents to localhost:{http_port}.

    Used by both local and bwrap backends — agents are always
    reachable on the loopback interface when running on the same host.
    """

    def __init__(self, agent_registry: AgentRegistry) -> None:
        self._agents = agent_registry

    async def resolve(self, agent_id: str) -> str:
        agent = await self._agents.get(agent_id)
        if not agent:
            raise KeyError(f"Agent not found: {agent_id}")
        return f"http://127.0.0.1:{agent.http_port}"
