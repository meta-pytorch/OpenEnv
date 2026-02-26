# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""AgentBus entry parsing and polling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# The AgentBus server caps a single poll response at 64 entries.
_SERVER_PAGE_SIZE = 64
_MAX_ENTRIES_PER_REQUEST = 1000


@dataclass
class BusEntry:
    """Typed representation of a bus log entry."""

    log_position: int
    type: str  # "intention", "vote", "commit", "abort", etc.
    content: str | None = None
    intention_id: int | None = None
    reason: str | None = None
    boolean_vote: bool | None = None
    probability_vote: float | None = None
    model: str | None = None
    policy: str | None = None
    prompt_override: str | None = None


def _proto_to_bus_entry(entry: Any) -> BusEntry:
    """Convert a protobuf ``BusEntry`` to a ``BusEntry`` dataclass."""
    log_position = entry.header.log_position
    payload_type = entry.payload.WhichOneof("payload")
    entry_type = payload_type or "unknown"

    kwargs: dict[str, Any] = {
        "log_position": log_position,
        "type": entry_type,
    }

    if payload_type == "intention":
        intention = entry.payload.intention
        kind = intention.WhichOneof("intention")
        if kind == "string_intention":
            kwargs["content"] = intention.string_intention

    elif payload_type == "vote":
        vote = entry.payload.vote
        kwargs["intention_id"] = vote.intention_id
        vote_kind = vote.abstract_vote.WhichOneof("vote_type")
        if vote_kind == "boolean_vote":
            kwargs["boolean_vote"] = vote.abstract_vote.boolean_vote
        elif vote_kind == "probability_vote":
            kwargs["probability_vote"] = vote.abstract_vote.probability_vote
        info_kind = vote.info.WhichOneof("vote_info")
        if info_kind == "external_llm_vote_info":
            kwargs["reason"] = vote.info.external_llm_vote_info.reason
            kwargs["model"] = vote.info.external_llm_vote_info.model

    elif payload_type == "commit":
        kwargs["intention_id"] = entry.payload.commit.intention_id
        kwargs["reason"] = entry.payload.commit.reason

    elif payload_type == "abort":
        kwargs["intention_id"] = entry.payload.abort.intention_id
        kwargs["reason"] = entry.payload.abort.reason

    elif payload_type == "decider_policy":
        kwargs["policy"] = str(entry.payload.decider_policy)

    elif payload_type == "inference_input":
        ii = entry.payload.inference_input
        kind = ii.WhichOneof("inference_input")
        if kind == "string_inference_input":
            kwargs["content"] = ii.string_inference_input

    elif payload_type == "inference_output":
        io = entry.payload.inference_output
        kind = io.WhichOneof("inference_output")
        if kind == "string_inference_output":
            kwargs["content"] = io.string_inference_output

    elif payload_type == "action_output":
        ao = entry.payload.action_output
        kwargs["intention_id"] = ao.intention_id
        kind = ao.WhichOneof("action_output")
        if kind == "string_action_output":
            kwargs["content"] = ao.string_action_output

    elif payload_type == "agent_input":
        ai = entry.payload.agent_input
        kind = ai.WhichOneof("agent_input")
        if kind == "string_agent_input":
            kwargs["content"] = ai.string_agent_input

    elif payload_type == "agent_output":
        ao = entry.payload.agent_output
        kind = ao.WhichOneof("agent_output")
        if kind == "string_agent_output":
            kwargs["content"] = ao.string_agent_output

    elif payload_type == "voter_policy":
        kwargs["prompt_override"] = entry.payload.voter_policy.prompt_override

    elif payload_type == "control":
        ctrl = entry.payload.control
        kind = ctrl.WhichOneof("control")
        if kind == "agent_input":
            kwargs["content"] = ctrl.agent_input

    else:
        if payload_type is not None:
            logger.warning(
                "Unhandled agentbus payload type: %s (log_position=%d)",
                payload_type,
                log_position,
            )

    return BusEntry(**kwargs)


async def poll_bus_entries(
    host: str,
    port: int,
    bus_id: str,
    *,
    filter_types: list[int] | None = None,
    max_entries: int = _MAX_ENTRIES_PER_REQUEST,
    follow: bool = False,
    timeout: float = 300.0,
) -> AsyncIterator[BusEntry]:
    """Async generator that polls entries from an AgentBus.

    Connects to the gRPC server at ``host:port``, polls entries for
    ``bus_id`` from position 0, and yields ``BusEntry`` objects.

    Args:
        host: gRPC server hostname.
        port: gRPC server port.
        bus_id: AgentBus bus identifier.
        filter_types: Optional list of ``SelectivePollType`` enum values.
        max_entries: Max entries per poll request.
        follow: If True, keep polling for new entries until *timeout*.
            If False (default), drain all current entries and stop.
        timeout: Seconds to keep polling in follow mode (default 300).

    Yields:
        ``BusEntry`` objects in log-position order.
    """
    import asyncio

    import grpc  # pyre-ignore[21]
    from agentbus.proto.agent_bus_pb2 import (  # pyre-ignore[21]
        PayloadTypeFilter,
        PollRequest,
    )
    from agentbus.proto.agent_bus_pb2_grpc import AgentBusServiceStub  # pyre-ignore[21]

    endpoint = f"{host}:{port}"
    deadline = asyncio.get_event_loop().time() + timeout if follow else None

    async with grpc.aio.insecure_channel(
        endpoint, options=[("grpc.enable_http_proxy", 0)]
    ) as channel:
        stub = AgentBusServiceStub(channel)
        start_pos = 0

        while True:
            kwargs: dict[str, Any] = {
                "agent_bus_id": bus_id,
                "start_log_position": start_pos,
                "max_entries": max_entries,
            }
            if filter_types is not None:
                kwargs["filter"] = PayloadTypeFilter(  # pyre-fixme[16]
                    payload_types=filter_types,
                )

            resp = await stub.Poll(PollRequest(**kwargs))  # pyre-fixme[16]

            for proto_entry in resp.entries:
                bus_entry = _proto_to_bus_entry(proto_entry)
                start_pos = proto_entry.header.log_position + 1
                yield bus_entry

            if resp.complete or len(resp.entries) == 0:
                if not follow or (
                    deadline is not None and asyncio.get_event_loop().time() >= deadline
                ):
                    break
                await asyncio.sleep(1.0)
