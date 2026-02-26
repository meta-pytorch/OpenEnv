# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

from typing import Any, Dict, List

from agentbus.agentbus_abc import AgentBusABC
from agentbus.proto.agent_bus_pb2 import (
    BusEntry,
    Header,
    Payload,
    PollRequest,
    PollResponse,
    ProposeRequest,
    ProposeResponse,
    SelectivePollType,
)


# Maximum number of entries that can be returned in a single poll request
MAX_POLL_ENTRIES: int = 64


class InMemoryAgentBus(AgentBusABC):
    """Simple in-memory implementation of AgentBus for testing.

    TODO: This class is untested for now, until integrate it with the existing
    rust tests.
    """

    # Map payload field names to SelectivePollType
    # Using Any because Python 3.10's typing doesn't support protobuf enums (EnumTypeWrapper instances)
    _PAYLOAD_TYPE_MAP: Dict[str, Any] = {
        "intention": SelectivePollType.INTENTION,
        "vote": SelectivePollType.VOTE,
        "decider_policy": SelectivePollType.DECIDER_POLICY,
        "voter_policy": SelectivePollType.VOTER_POLICY,
        "commit": SelectivePollType.COMMIT,
        "abort": SelectivePollType.ABORT,
        "control": SelectivePollType.CONTROL,
        "inference_input": SelectivePollType.INFERENCE_INPUT,
        "inference_output": SelectivePollType.INFERENCE_OUTPUT,
        "action_output": SelectivePollType.ACTION_OUTPUT,
        "agent_input": SelectivePollType.AGENT_INPUT,
        "agent_output": SelectivePollType.AGENT_OUTPUT,
    }

    def __init__(self) -> None:
        # Map from agentBusId to list of payloads
        self._buses: Dict[int, List[Payload]] = {}

    async def propose(self, request: ProposeRequest) -> ProposeResponse:
        """Override of AgentBusABC.propose - see base class for documentation."""
        agent_bus_id = request.agent_bus_id
        payload = request.payload

        # Get or create the bus
        if agent_bus_id not in self._buses:
            self._buses[agent_bus_id] = []

        bus = self._buses[agent_bus_id]
        position = len(bus)
        bus.append(payload)

        return ProposeResponse(log_position=position)

    async def poll(self, request: PollRequest) -> PollResponse:
        """Override of AgentBusABC.poll - see base class for documentation."""
        agent_bus_id = request.agent_bus_id
        start_position = request.start_log_position
        max_entries = min(request.max_entries, MAX_POLL_ENTRIES)
        payload_filter = request.filter if request.HasField("filter") else None

        # Get the bus or return empty
        bus = self._buses.get(agent_bus_id)
        if bus is None:
            return PollResponse(entries=[], complete=True)

        # Build the result
        entries: List[BusEntry] = []
        complete = True

        # Get slice of bus starting from start_position
        if start_position < len(bus):
            bus_slice = bus[start_position:]

            for i, payload in enumerate(bus_slice):
                # Apply payload type filter if present
                if payload_filter is not None:
                    payload_types = set(payload_filter.payload_types)
                    # Empty filter means match nothing
                    if len(payload_types) == 0:
                        break

                    payload_type = self._get_payload_type(payload)
                    if payload_type is None or payload_type not in payload_types:
                        continue  # Skip entries that don't match filter

                if len(entries) >= max_entries:
                    # More entries available
                    complete = False
                    break

                entry = BusEntry(
                    header=Header(log_position=start_position + i),
                    payload=payload,
                )
                entries.append(entry)

        return PollResponse(entries=entries, complete=complete)

    def _get_payload_type(self, payload: Payload) -> Any:
        """Map a Payload to its SelectivePollType using WhichOneof.

        Returns Any because Python 3.10's typing doesn't support protobuf enums.
        """
        payload_field = payload.WhichOneof("payload")
        if payload_field:
            return self._PAYLOAD_TYPE_MAP.get(payload_field)
        return None
