# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""BusService - read-only service for querying AgentBus entries."""

import logging
from typing import AsyncIterator

from .storage.registry import AgentRegistry

logger = logging.getLogger(__name__)

# Re-export so existing importers of `from .bus import BusEntry` etc. still work.
from .bus_entries import _proto_to_bus_entry, BusEntry  # noqa: E402, F401


class BusService:
    """Read-only service for querying AgentBus entries.

    Looks up agent bus config from the registry, connects to the
    bus gRPC server, and streams entries.
    """

    def __init__(self, agent_registry: AgentRegistry) -> None:
        self._registry = agent_registry

    async def entries(
        self,
        agent_id: str,
        *,
        follow: bool = False,
        timeout: float = 300.0,
        filter_types: list[str] | None = None,
    ) -> AsyncIterator[BusEntry]:
        """Yield bus entries for an agent.

        Looks up the agent's bus config from the registry, connects to
        the bus gRPC server, and polls entries.

        follow=False: drain all current entries and stop.
        follow=True: keep polling for new entries until timeout (default 300s).
        filter_types: optional list of type strings (e.g. ["intention", "vote"]).

        Raises:
            KeyError: If agent not found in registry.
            ValueError: If agent has no agentbus configured.
        """
        agent = await self._registry.get(agent_id)
        if agent is None:
            raise KeyError(f"Agent not found: {agent_id}")
        if agent.agentbus is None:
            raise ValueError(f"Agent {agent_id} has no agentbus configured")
        agentbus_cfg = agent.agentbus

        # Lazy import so agentbus package remains optional
        from .bus_config import parse_agentbus_url

        parsed = parse_agentbus_url(agentbus_cfg.url)
        host = parsed.get("host", "localhost")
        port = parsed.get("port")
        if port is None:
            raise ValueError(
                f"Cannot connect to agentbus for agent {agent_id}: "
                f"no port in URL '{agentbus_cfg.url}'"
            )

        bus_id = f"{agent.name}.{agent.id}"

        # Map string filter types to SelectivePollType enum values
        resolved_filter_types: list[int] | None = None
        if filter_types is not None:
            from agentbus.proto.agent_bus_pb2 import (  # pyre-ignore[21]
                SelectivePollType,
            )

            resolved_filter_types = []
            for name in filter_types:
                enum_name = name.upper()
                if hasattr(SelectivePollType, enum_name):  # pyre-ignore[16]
                    resolved_filter_types.append(
                        getattr(SelectivePollType, enum_name)  # pyre-ignore[16]
                    )

        from .bus_entries import poll_bus_entries

        async for entry in poll_bus_entries(
            host,
            port,
            bus_id,
            filter_types=resolved_filter_types,
            follow=follow,
            timeout=timeout,
        ):
            yield entry
