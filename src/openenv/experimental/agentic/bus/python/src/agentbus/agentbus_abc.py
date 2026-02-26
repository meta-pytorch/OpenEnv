# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

from abc import ABC, abstractmethod

from agentbus.proto.agent_bus_pb2 import (
    PollRequest,
    PollResponse,
    ProposeRequest,
    ProposeResponse,
)


class AgentBusABC(ABC):
    """Abstract base class for AgentBus implementations."""

    @abstractmethod
    async def propose(self, request: ProposeRequest) -> ProposeResponse:
        """Propose a payload to the agent bus.

        Args:
            request: ProposeRequest containing agent_bus_id and payload

        Returns:
            ProposeResponse with the log position where the entry was stored
        """
        ...

    @abstractmethod
    async def poll(self, request: PollRequest) -> PollResponse:
        """Poll for entries in the agent bus.

        Args:
            request: PollRequest containing agent_bus_id, start_log_position, max_entries, and optional filter (PayloadTypeFilter)

        Returns:
            PollResponse with entries and complete flag
        """
        ...
