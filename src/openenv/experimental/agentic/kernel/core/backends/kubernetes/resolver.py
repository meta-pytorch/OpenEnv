# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""KubernetesResolver - resolves agents to localhost via port-forward."""

from ...storage.registry import AgentRegistry


class KubernetesResolver:
    """Resolves agents to ``http://127.0.0.1:{agent.http_port}``.

    Because the spawner sets up ``kubectl port-forward`` for each agent,
    they are reachable on localhost — same as LocalResolver.  This is
    kept as a separate class for future evolution (e.g., switching to
    ClusterIP DNS when the kernel itself runs inside the cluster).
    """

    def __init__(self, agent_registry: AgentRegistry) -> None:
        self._agents = agent_registry

    async def resolve(self, agent_id: str) -> str:
        agent = await self._agents.get(agent_id)
        if not agent:
            raise KeyError(f"Agent not found: {agent_id}")
        return f"http://127.0.0.1:{agent.http_port}"
