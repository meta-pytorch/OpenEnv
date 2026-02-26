# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
import unittest
import unittest.mock as mock

from agentbus.agentbus_client.agentbus_client import AgentBusClient
from agentbus.agentbus_client.agentbus_config import AgentBusConfig
from agentbus.agentbus_client.agentbus_errors import AgentBusClientError
from agentbus.proto.agent_bus_pb2 import (
    Abort,
    BusEntry,
    Commit,
    DeciderPolicy,
    Header,
    Intention,
    Payload,
    PollResponse,
    ProposeResponse,
)


def create_intention_entry(log_position: int, code: str) -> BusEntry:
    return BusEntry(
        header=Header(log_position=log_position),
        payload=Payload(intention=Intention(string_intention=code)),
    )


def create_commit_entry(log_position: int, intention_id: int) -> BusEntry:
    return BusEntry(
        header=Header(log_position=log_position),
        payload=Payload(commit=Commit(intention_id=intention_id)),
    )


def create_abort_entry(
    log_position: int, intention_id: int, reason: str = ""
) -> BusEntry:
    return BusEntry(
        header=Header(log_position=log_position),
        payload=Payload(abort=Abort(intention_id=intention_id, reason=reason)),
    )


class AgentBusClientTestBase(unittest.IsolatedAsyncioTestCase):
    """Base test class that sets up a mock AgentBusClient with a fake stub."""

    def setUp(self) -> None:
        super().setUp()
        self.config = AgentBusConfig(host="localhost", port=9999, bus_id="bus-0")
        self.stub_mock = mock.AsyncMock()

        channel_patcher = mock.patch(
            "agentbus.agentbus_client.agentbus_client._get_grpc_channel"
        )
        self.mock_get_channel = channel_patcher.start()
        self.mock_channel = mock.MagicMock()
        self.mock_channel.close = mock.AsyncMock()
        self.mock_get_channel.return_value = self.mock_channel
        self.addCleanup(channel_patcher.stop)

        stub_patcher = mock.patch(
            "agentbus.agentbus_client.agentbus_client.AgentBusServiceStub",
            return_value=self.stub_mock,
        )
        stub_patcher.start()
        self.addCleanup(stub_patcher.stop)

    def _make_client(self) -> AgentBusClient:
        return AgentBusClient(self.config)


class GetCommittedIntentionsTest(AgentBusClientTestBase):
    async def test_single_poll(self) -> None:
        client = self._make_client()
        entries = [
            create_intention_entry(1, "print('hello')"),
            create_intention_entry(2, "print('world')"),
            create_commit_entry(3, 1),
            create_commit_entry(4, 2),
        ]
        self.stub_mock.Poll.return_value = PollResponse(entries=entries, complete=True)
        result = await client.get_committed_intentions(
            logger=logging.getLogger(__name__)
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "print('hello')")
        self.assertEqual(result[1], "print('world')")
        self.stub_mock.Poll.assert_called_once()

    async def test_multiple_polls(self) -> None:
        client = self._make_client()
        entries1 = [
            create_intention_entry(1, "code1"),
            create_commit_entry(2, 1),
        ]
        entries2 = [
            create_intention_entry(3, "code3"),
            create_commit_entry(4, 3),
        ]
        self.stub_mock.Poll.side_effect = [
            PollResponse(entries=entries1, complete=False),
            PollResponse(entries=entries2, complete=True),
        ]
        result = await client.get_committed_intentions(
            logger=logging.getLogger(__name__)
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "code1")
        self.assertEqual(result[1], "code3")
        self.assertEqual(self.stub_mock.Poll.call_count, 2)

    async def test_max_exceeded(self) -> None:
        client = self._make_client()
        max_intentions = 2
        num_intentions = 3
        entries = [
            create_intention_entry(i, f"code{i}") for i in range(num_intentions)
        ] + [create_commit_entry(i + num_intentions, i) for i in range(num_intentions)]
        self.stub_mock.Poll.return_value = PollResponse(entries=entries, complete=True)
        with self.assertRaises(Exception) as context:
            await client.get_committed_intentions(
                logger=logging.getLogger(__name__),
                max_intentions=max_intentions,
            )
        self.assertIn("max_intentions", str(context.exception))

    async def test_intention_id_not_found(self) -> None:
        client = self._make_client()
        entries = [
            create_intention_entry(1, "print('hello')"),
            create_commit_entry(2, 1),
            create_commit_entry(3, 5),
        ]
        self.stub_mock.Poll.return_value = PollResponse(entries=entries, complete=True)
        with self.assertRaises(AgentBusClientError) as context:
            await client.get_committed_intentions(logger=logging.getLogger(__name__))
        self.assertIn("Committed intention ids [5] not found", str(context.exception))


class LogIntentionTest(AgentBusClientTestBase):
    async def test_returns_log_position(self) -> None:
        client = self._make_client()
        code = "print('hello world')"
        intent_log_pos = 42
        self.stub_mock.Propose.return_value = ProposeResponse(
            log_position=intent_log_pos
        )
        result = await client.log_intention(
            code=code, logger=logging.getLogger(__name__)
        )
        self.assertEqual(result, intent_log_pos)
        self.stub_mock.Propose.assert_called_once()

    async def test_raises_on_error(self) -> None:
        client = self._make_client()
        code = "print('hello world')"
        error_message = "Connection refused"
        self.stub_mock.Propose.side_effect = Exception(error_message)
        with self.assertRaises(Exception) as context:
            await client.log_intention(code=code, logger=logging.getLogger(__name__))
        self.assertIn(error_message, str(context.exception))


class WaitForSafetyDecisionTest(AgentBusClientTestBase):
    async def test_approved(self) -> None:
        client = self._make_client()
        intent_log_pos = 42
        entries_with_commit = [
            create_commit_entry(43, intent_log_pos),
        ]
        self.stub_mock.Poll.return_value = PollResponse(
            entries=entries_with_commit, complete=True
        )
        decision = await client.wait_for_safety_decision(
            intention_id=intent_log_pos,
            logger=logging.getLogger(__name__),
        )
        self.assertTrue(decision.approved)
        self.stub_mock.Poll.assert_called_once()

    async def test_rejected(self) -> None:
        client = self._make_client()
        intent_log_pos = 42
        entries = [
            create_abort_entry(43, intent_log_pos, "Dangerous operation detected"),
        ]
        self.stub_mock.Poll.return_value = PollResponse(entries=entries, complete=True)
        decision = await client.wait_for_safety_decision(
            intention_id=intent_log_pos,
            logger=logging.getLogger(__name__),
        )
        self.assertFalse(decision.approved)
        self.stub_mock.Poll.assert_called_once()

    async def test_multiple_polls(self) -> None:
        client = self._make_client()
        intent_log_pos = 42
        entries_with_commit = [
            create_commit_entry(43, intent_log_pos),
        ]
        self.stub_mock.Poll.side_effect = [
            PollResponse(entries=[], complete=True),
            PollResponse(entries=[], complete=True),
            PollResponse(entries=entries_with_commit, complete=True),
        ]
        decision = await client.wait_for_safety_decision(
            intention_id=intent_log_pos,
            logger=logging.getLogger(__name__),
        )
        self.assertTrue(decision.approved)
        self.assertEqual(self.stub_mock.Poll.call_count, 3)

    async def test_timeout(self) -> None:
        client = self._make_client()
        intent_log_pos = 42
        max_poll_attempts = 3
        self.stub_mock.Poll.return_value = PollResponse(entries=[], complete=True)
        decision = await client.wait_for_safety_decision(
            intention_id=intent_log_pos,
            logger=logging.getLogger(__name__),
            max_poll_attempts=max_poll_attempts,
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "Safety check timed out")
        self.assertEqual(self.stub_mock.Poll.call_count, max_poll_attempts)


class LogLLMInferenceTest(AgentBusClientTestBase):
    async def test_log_llm_input(self) -> None:
        client = self._make_client()
        input_str = '{"messages": [{"role": "user", "content": "hello"}]}'
        log_pos = 42
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=log_pos)

        result = await client.log_llm_input(input_str)

        self.assertEqual(result, log_pos)
        self.stub_mock.Propose.assert_called_once()
        call_args = self.stub_mock.Propose.call_args[0][0]
        self.assertEqual(call_args.agent_bus_id, "bus-0")
        self.assertEqual(
            call_args.payload.inference_input.string_inference_input, input_str
        )

    async def test_log_llm_output(self) -> None:
        client = self._make_client()
        output_str = '{"message_dict": {"role": "assistant"}, "result_dict": {}}'
        log_pos = 43
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=log_pos)

        result = await client.log_llm_output(output_str)

        self.assertEqual(result, log_pos)
        self.stub_mock.Propose.assert_called_once()
        call_args = self.stub_mock.Propose.call_args[0][0]
        self.assertEqual(call_args.agent_bus_id, "bus-0")
        self.assertEqual(
            call_args.payload.inference_output.string_inference_output, output_str
        )

    async def test_log_llm_input_raises_on_error(self) -> None:
        client = self._make_client()
        self.stub_mock.Propose.side_effect = Exception("Connection error")

        with self.assertRaises(Exception) as context:
            await client.log_llm_input('{"test": "data"}')

        self.assertIn("Connection error", str(context.exception))

    async def test_log_llm_output_raises_on_error(self) -> None:
        client = self._make_client()
        self.stub_mock.Propose.side_effect = Exception("Connection error")

        with self.assertRaises(Exception) as context:
            await client.log_llm_output('{"test": "data"}')

        self.assertIn("Connection error", str(context.exception))


class LogAgentAndActionTest(AgentBusClientTestBase):
    async def test_log_agent_input(self) -> None:
        client = self._make_client()
        input_str = "What is the weather today?"
        log_pos = 42
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=log_pos)

        result = await client.log_agent_input(input_str)

        self.assertEqual(result, log_pos)
        self.stub_mock.Propose.assert_called_once()
        call_args = self.stub_mock.Propose.call_args[0][0]
        self.assertEqual(call_args.agent_bus_id, "bus-0")
        self.assertEqual(call_args.payload.agent_input.string_agent_input, input_str)

    async def test_log_agent_output(self) -> None:
        client = self._make_client()
        output_str = "The weather is sunny."
        log_pos = 43
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=log_pos)

        result = await client.log_agent_output(output_str)

        self.assertEqual(result, log_pos)
        self.stub_mock.Propose.assert_called_once()
        call_args = self.stub_mock.Propose.call_args[0][0]
        self.assertEqual(call_args.agent_bus_id, "bus-0")
        self.assertEqual(call_args.payload.agent_output.string_agent_output, output_str)

    async def test_log_action_output(self) -> None:
        client = self._make_client()
        output_str = '{"result": "success"}'
        intention_id = 10
        log_pos = 44
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=log_pos)

        result = await client.log_action_output(output_str, intention_id)

        self.assertEqual(result, log_pos)
        self.stub_mock.Propose.assert_called_once()
        call_args = self.stub_mock.Propose.call_args[0][0]
        self.assertEqual(call_args.agent_bus_id, "bus-0")
        self.assertEqual(call_args.payload.action_output.intention_id, intention_id)
        self.assertEqual(
            call_args.payload.action_output.string_action_output, output_str
        )

    async def test_log_action_output_empty_output(self) -> None:
        client = self._make_client()
        intention_id = 42
        log_pos = 44
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=log_pos)

        result = await client.log_action_output("", intention_id)

        self.assertEqual(result, log_pos)
        self.stub_mock.Propose.assert_called_once()
        call_args = self.stub_mock.Propose.call_args[0][0]
        self.assertEqual(call_args.payload.action_output.string_action_output, "")

    async def test_log_agent_input_raises_on_error(self) -> None:
        client = self._make_client()
        self.stub_mock.Propose.side_effect = Exception("Connection error")

        with self.assertRaises(Exception) as context:
            await client.log_agent_input("test input")

        self.assertIn("Connection error", str(context.exception))

    async def test_log_agent_output_raises_on_error(self) -> None:
        client = self._make_client()
        self.stub_mock.Propose.side_effect = Exception("Connection error")

        with self.assertRaises(Exception) as context:
            await client.log_agent_output("test output")

        self.assertIn("Connection error", str(context.exception))

    async def test_log_action_output_raises_on_error(self) -> None:
        client = self._make_client()
        self.stub_mock.Propose.side_effect = Exception("Connection error")

        with self.assertRaises(Exception) as context:
            await client.log_action_output("test output", 10)

        self.assertIn("Connection error", str(context.exception))


class ProposeDeciderPolicyTest(AgentBusClientTestBase):
    async def test_set_decider_policy_returns_log_position(self) -> None:
        client = self._make_client()
        log_pos = 42
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=log_pos)

        result = await client.set_decider_policy(DeciderPolicy.FIRST_BOOLEAN_WINS)

        self.assertEqual(result, log_pos)
        self.stub_mock.Propose.assert_called_once()

    async def test_set_decider_policy_first_boolean_wins(self) -> None:
        client = self._make_client()
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=1)

        await client.set_decider_policy(DeciderPolicy.FIRST_BOOLEAN_WINS)

        call_args = self.stub_mock.Propose.call_args[0][0]
        self.assertEqual(call_args.agent_bus_id, "bus-0")
        self.assertEqual(
            call_args.payload.decider_policy, DeciderPolicy.FIRST_BOOLEAN_WINS
        )

    async def test_set_decider_policy_on_by_default(self) -> None:
        client = self._make_client()
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=1)

        await client.set_decider_policy(DeciderPolicy.ON_BY_DEFAULT)

        call_args = self.stub_mock.Propose.call_args[0][0]
        self.assertEqual(call_args.agent_bus_id, "bus-0")
        self.assertEqual(call_args.payload.decider_policy, DeciderPolicy.ON_BY_DEFAULT)


class ChannelLifecycleTest(AgentBusClientTestBase):
    async def test_channel_created_once_on_enter(self) -> None:
        client = self._make_client()
        self.mock_get_channel.assert_called_once_with(self.config)
        # Make multiple calls — channel should not be recreated
        self.stub_mock.Propose.return_value = ProposeResponse(log_position=1)
        await client.log_llm_input("a")
        await client.log_llm_output("b")
        self.mock_get_channel.assert_called_once()

    async def test_channel_closed_on_close(self) -> None:
        client = self._make_client()
        await client.close()
        self.mock_channel.close.assert_called_once()
