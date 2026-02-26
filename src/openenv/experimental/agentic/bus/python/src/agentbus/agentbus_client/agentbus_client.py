# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

import grpc
from agentbus.agentbus_client.agentbus_config import AgentBusConfig
from agentbus.agentbus_client.agentbus_errors import AgentBusClientError
from agentbus.proto.agent_bus_pb2 import (
    ActionOutput,
    AgentInput,
    AgentOutput,
    BusEntry,
    InferenceInput,
    InferenceOutput,
    Intention,
    Payload,
    PayloadTypeFilter,
    PollRequest,
    ProposeRequest,
    SelectivePollType,
)
from agentbus.proto.agent_bus_pb2_grpc import AgentBusServiceStub

# NOTE: AgentBus server has a server limit of returning 64 entries in one poll
# request now https://fburl.com/code/y3f5gc6a
# so the max_entries value in the poll request isn't really effective if it's larger than 64.
# We will switch to streaming API soon so all of these will be gone soon.
MAX_ENTRIES_IN_POLL_REQUEST = 1000


@dataclass(frozen=True)
class AgentBusDecision:
    """Represents a decision from AgentBus about code safety."""

    approved: bool
    reason: str
    log_position: int | None = (
        None  # The intention's log position (for linking ActionOutput)
    )


async def _propose_intention(
    client: Any,
    code: str,
    bus_id: str,
    logger: logging.Logger,
) -> int:
    """Propose code as an intention to AgentBus and return the log position."""
    intention = Intention(string_intention=code)
    payload = Payload(intention=intention)
    propose_request = ProposeRequest(
        agent_bus_id=bus_id,
        payload=payload,
    )
    propose_response = await client.Propose(propose_request)
    intent_log_pos = propose_response.log_position
    logger.debug(
        f"Intention proposed at log position {intent_log_pos}, waiting for decision..."
    )
    return intent_log_pos


def _check_poll_entries_for_decision(
    entries: list[BusEntry], intent_log_pos: int, logger: logging.Logger
) -> Optional[AgentBusDecision]:
    """
    Check poll entries for commit/abort decisions matching our intention.

    Returns:
        AgentBusDecision with approved=True if COMMIT found,
        AgentBusDecision with approved=False if ABORT found,
        None if no decision found
    """
    for entry in entries:
        payload_type = entry.payload.WhichOneof("payload")

        if (
            payload_type == "commit"
            and (commit_value := entry.payload.commit)
            and commit_value.intention_id == intent_log_pos
        ):
            reason = commit_value.reason
            logger.debug(f"Code approved by safety check (COMMIT): {reason}")
            return AgentBusDecision(approved=True, reason=reason)
        elif (
            payload_type == "abort"
            and (abort_value := entry.payload.abort)
            and abort_value.intention_id == intent_log_pos
        ):
            reason = abort_value.reason
            logger.debug(f"Code blocked by safety check (ABORT): {reason}")
            return AgentBusDecision(approved=False, reason=reason)

    return None


async def _poll_for_decision(
    client: Any,
    bus_id: str,
    intent_log_pos: int,
    max_poll_attempts: int,
    logger: logging.Logger,
) -> AgentBusDecision:
    """
    Poll AgentBus for a decision on the proposed intention.

    Returns:
        AgentBusDecision with approved=True if code is approved,
        approved=False if blocked or timeout occurs
    """
    poll_attempt = 0
    while poll_attempt < max_poll_attempts:
        poll_request = PollRequest(
            agent_bus_id=bus_id,
            start_log_position=intent_log_pos + 1,  # Start polling AFTER the intention
            max_entries=MAX_ENTRIES_IN_POLL_REQUEST,  # TODO: figure how to do this smartly later.
            filter=PayloadTypeFilter(
                payload_types=[SelectivePollType.COMMIT, SelectivePollType.ABORT]
            ),
        )

        poll_response = await client.Poll(poll_request)

        decision = _check_poll_entries_for_decision(
            poll_response.entries, intent_log_pos, logger
        )
        if decision is not None:
            return decision

        await asyncio.sleep(1.0)
        poll_attempt += 1

    logger.warning(
        f"AgentBus safety check timed out after {max_poll_attempts} attempts, blocking execution"
    )
    return AgentBusDecision(approved=False, reason="Safety check timed out")


async def _get_committed_intentions_from_client(
    client: Any,
    bus_id: str,
    max_intentions: int,
    logger: logging.Logger,
) -> list[Intention]:
    """Get all the committed intentions from agentbus for the specified bus_id"""
    intentions_by_id = {}
    committed_intention_ids = []

    start_log_pos = 0
    while True:
        poll_request = PollRequest(
            agent_bus_id=bus_id,
            start_log_position=start_log_pos,
            max_entries=MAX_ENTRIES_IN_POLL_REQUEST,
            filter=PayloadTypeFilter(
                payload_types=[SelectivePollType.INTENTION, SelectivePollType.COMMIT]
            ),
        )
        poll_response = await client.Poll(poll_request)

        for entry in poll_response.entries:
            payload_type = entry.payload.WhichOneof("payload")
            if payload_type == "intention":
                intentions_by_id[entry.header.log_position] = entry.payload.intention
            elif payload_type == "commit":
                intention_id = entry.payload.commit.intention_id
                committed_intention_ids.append(intention_id)
            else:
                logger.warning(f"Unknown payload type: {payload_type}")
            # next poll request should start from the next log position
            start_log_pos = entry.header.log_position + 1

        if len(committed_intention_ids) > max_intentions:
            raise Exception(
                f"max_intentions ({max_intentions}) was "
                "too small to cover all the committed intentions. Try increasing it."
            )
        if poll_response.complete:
            break

    missing_ids = [id for id in committed_intention_ids if id not in intentions_by_id]
    if missing_ids:
        raise AgentBusClientError(
            f"Committed intention ids {missing_ids} not found in intention "
            "entries. This indicates a data inconsistency in the AgentBus log."
        )
    return [intentions_by_id[id] for id in committed_intention_ids]


def _get_grpc_channel(
    config: AgentBusConfig,
) -> grpc.aio.Channel:
    """
    Get a gRPC channel based on the configuration.

    Args:
        config: AgentBusConfig with host and port

    Returns:
        gRPC async channel
    """
    endpoint = f"{config.host}:{config.port}"
    return grpc.aio.insecure_channel(endpoint, options=[("grpc.enable_http_proxy", 0)])


class AgentBusClient:
    """gRPC client that holds a single channel across calls.

    Usage:
        client = AgentBusClient(config)
        await client.log_llm_input(input_str)
        intention_id = await client.log_intention(code)
        decision = await client.wait_for_safety_decision(intention_id)
        await client.close()
    """

    def __init__(self, config: AgentBusConfig) -> None:
        self._config = config
        self._channel: grpc.aio.Channel = _get_grpc_channel(config)
        self._stub: AgentBusServiceStub = AgentBusServiceStub(self._channel)

    async def close(self) -> None:
        """Close the underlying gRPC channel."""
        await self._channel.close()

    async def log_intention(
        self,
        code: str,
        logger: Optional[logging.Logger] = None,
    ) -> int:
        """
        Log code as an intention to AgentBus.

        Args:
            code: The code to log as an intention
            logger: Optional logger for debugging

        Returns:
            Log position (intention_id) of the proposed intention
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            intent_log_pos = await _propose_intention(
                self._stub, code, self._config.bus_id, logger
            )
            return intent_log_pos
        except Exception as e:
            logger.error(f"AgentBus intention logging failed: {e}")
            raise

    async def wait_for_safety_decision(
        self,
        intention_id: int,
        logger: Optional[logging.Logger] = None,
        max_poll_attempts: int = 30,
    ) -> AgentBusDecision:
        """
        Wait for a safety decision (commit/abort) for a previously logged intention.

        Args:
            intention_id: The log position of the intention to check
            logger: Optional logger for debugging
            max_poll_attempts: Maximum number of polling attempts (default: 30)

        Returns:
            AgentBusDecision with approved=True if code is approved,
            approved=False if blocked or timeout occurs.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            decision = await _poll_for_decision(
                self._stub, self._config.bus_id, intention_id, max_poll_attempts, logger
            )
            return AgentBusDecision(
                approved=decision.approved,
                reason=decision.reason,
            )
        except Exception as e:
            logger.error(f"AgentBus safety check failed: {e}")
            return AgentBusDecision(
                approved=False,
                reason=f"AgentBus safety check failed: {e}",
            )

    async def get_committed_intentions(
        self,
        logger: Optional[logging.Logger] = None,
        max_intentions: int = 5000,
    ) -> list[str]:
        """
        Get all committed intentions from AgentBus.

        Args:
            logger: Optional logger instance for logging messages
            max_intentions: max number of committed intentions to return. Set it to a value
                big enough to cover all the committed intentions, otherwise error will be thrown.

        Returns:
            List of strings containing the content of all committed intentions
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        try:
            committed_intentions = await _get_committed_intentions_from_client(
                self._stub, self._config.bus_id, max_intentions, logger
            )
            if not all(
                intention.WhichOneof("intention") == "string_intention"
                for intention in committed_intentions
            ):
                raise AgentBusClientError("Got unsupported AgentBus intention type")
            return [intention.string_intention for intention in committed_intentions]
        except Exception as e:
            msg = f"Failed to get committed intentions from AgentBus: {e}"
            logger.error(msg)
            raise AgentBusClientError(msg)

    async def log_llm_input(
        self,
        input_str: str,
    ) -> int:
        """
        Log LLM inference input to AgentBus as an InferenceInput.

        Args:
            input_str: String representation of the LLM input

        Returns:
            Log position of the proposed InferenceInput
        """
        inference_input = InferenceInput(string_inference_input=input_str)
        payload = Payload(inference_input=inference_input)
        propose_request = ProposeRequest(
            agent_bus_id=self._config.bus_id,
            payload=payload,
        )
        propose_response = await self._stub.Propose(propose_request)
        return propose_response.log_position

    async def log_llm_output(
        self,
        output_str: str,
    ) -> int:
        """
        Log LLM inference output to AgentBus as an InferenceOutput.

        Args:
            output_str: String representation of the LLM output

        Returns:
            Log position of the proposed InferenceOutput
        """
        inference_output = InferenceOutput(string_inference_output=output_str)
        payload = Payload(inference_output=inference_output)
        propose_request = ProposeRequest(
            agent_bus_id=self._config.bus_id,
            payload=payload,
        )
        propose_response = await self._stub.Propose(propose_request)
        return propose_response.log_position

    async def log_agent_input(
        self,
        input_str: str,
    ) -> int:
        """
        Log agent input to AgentBus as an AgentInput.

        Args:
            input_str: String representation of the agent input

        Returns:
            Log position of the proposed AgentInput
        """
        agent_input = AgentInput(string_agent_input=input_str)
        payload = Payload(agent_input=agent_input)
        propose_request = ProposeRequest(
            agent_bus_id=self._config.bus_id,
            payload=payload,
        )
        propose_response = await self._stub.Propose(propose_request)
        return propose_response.log_position

    async def log_agent_output(
        self,
        output_str: str,
    ) -> int:
        """
        Log agent output to AgentBus as an AgentOutput.

        Args:
            output_str: String representation of the agent output

        Returns:
            Log position of the proposed AgentOutput
        """
        agent_output = AgentOutput(string_agent_output=output_str)
        payload = Payload(agent_output=agent_output)
        propose_request = ProposeRequest(
            agent_bus_id=self._config.bus_id,
            payload=payload,
        )
        propose_response = await self._stub.Propose(propose_request)
        return propose_response.log_position

    async def log_action_output(
        self,
        output_str: str,
        intention_id: int,
    ) -> int:
        """
        Log action output to AgentBus as an ActionOutput.

        Args:
            output_str: String representation of the action output
            intention_id: The intention ID to link this action output to

        Returns:
            Log position of the proposed ActionOutput
        """
        action_output = ActionOutput(
            intention_id=intention_id, string_action_output=output_str
        )
        payload = Payload(action_output=action_output)
        propose_request = ProposeRequest(
            agent_bus_id=self._config.bus_id,
            payload=payload,
        )
        propose_response = await self._stub.Propose(propose_request)
        return propose_response.log_position

    async def set_decider_policy(self, policy: int) -> int:
        """
        Propose the given decider policy to AgentBus.

        Args:
            policy: A DeciderPolicy enum value (e.g. DeciderPolicy.ON_BY_DEFAULT).

        Returns:
            Log position of the proposed DeciderPolicy
        """
        payload = Payload(decider_policy=policy)
        propose_request = ProposeRequest(
            agent_bus_id=self._config.bus_id,
            payload=payload,
        )
        propose_response = await self._stub.Propose(propose_request)
        return propose_response.log_position
