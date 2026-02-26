# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for BusService and BusEntry."""

import pytest
import pytest_asyncio
from agentic.kernel.core.bus import _proto_to_bus_entry, BusEntry, BusService
from agentic.kernel.core.config import Agent, AgentBusConfig, AgentState
from agentic.kernel.core.storage.registry import AgentRegistry


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry()


@pytest.fixture
def bus_service(registry: AgentRegistry) -> BusService:
    return BusService(registry)


@pytest_asyncio.fixture
async def agent_with_bus(registry: AgentRegistry) -> Agent:
    agent = Agent(
        id="test-agent-1",
        name="worker",
        team_id="team-1",
        agent_type="openclaw",
        image_id="",
        agentbus=AgentBusConfig(url="memory://9999"),
    )
    await registry.register(agent)
    return agent


@pytest_asyncio.fixture
async def agent_without_bus(registry: AgentRegistry) -> Agent:
    agent = Agent(
        id="test-agent-2",
        name="worker-no-bus",
        team_id="team-1",
        agent_type="openclaw",
        image_id="",
    )
    await registry.register(agent)
    return agent


# ── BusEntry dataclass tests ─────────────────────────────────────────


class TestBusEntry:
    def test_basic_fields(self) -> None:
        entry = BusEntry(log_position=0, type="intention", content="print('hello')")
        assert entry.log_position == 0
        assert entry.type == "intention"
        assert entry.content == "print('hello')"
        assert entry.intention_id is None
        assert entry.reason is None
        assert entry.boolean_vote is None

    def test_vote_fields(self) -> None:
        entry = BusEntry(
            log_position=5,
            type="vote",
            intention_id=3,
            boolean_vote=True,
            reason="Safe to execute",
            model="claude-sonnet-4-5",
        )
        assert entry.boolean_vote is True
        assert entry.intention_id == 3
        assert entry.reason == "Safe to execute"
        assert entry.model == "claude-sonnet-4-5"

    def test_commit_fields(self) -> None:
        entry = BusEntry(
            log_position=10,
            type="commit",
            intention_id=3,
            reason="All votes approve",
        )
        assert entry.type == "commit"
        assert entry.intention_id == 3
        assert entry.reason == "All votes approve"

    def test_abort_fields(self) -> None:
        entry = BusEntry(
            log_position=11,
            type="abort",
            intention_id=4,
            reason="Unsafe code detected",
        )
        assert entry.type == "abort"
        assert entry.intention_id == 4

    def test_probability_vote(self) -> None:
        entry = BusEntry(
            log_position=6,
            type="vote",
            intention_id=3,
            probability_vote=0.95,
        )
        assert entry.probability_vote == 0.95
        assert entry.boolean_vote is None

    def test_policy_fields(self) -> None:
        entry = BusEntry(
            log_position=1,
            type="decider_policy",
            policy="always_allow",
        )
        assert entry.policy == "always_allow"

    def test_voter_policy_fields(self) -> None:
        entry = BusEntry(
            log_position=2,
            type="voter_policy",
            prompt_override="Be strict about safety",
        )
        assert entry.prompt_override == "Be strict about safety"

    def test_defaults_are_none(self) -> None:
        entry = BusEntry(log_position=0, type="unknown")
        assert entry.content is None
        assert entry.intention_id is None
        assert entry.reason is None
        assert entry.boolean_vote is None
        assert entry.probability_vote is None
        assert entry.model is None
        assert entry.policy is None
        assert entry.prompt_override is None


# ── _proto_to_bus_entry tests ─────────────────────────────────────────


class _FakeOneof:
    """Helper to simulate protobuf WhichOneof behavior."""

    def __init__(self, field_name: str | None, **attrs: object) -> None:
        self._field_name = field_name
        for k, v in attrs.items():
            setattr(self, k, v)

    def WhichOneof(self, name: str) -> str | None:
        return self._field_name


class TestProtoToBusEntry:
    def test_intention_entry(self) -> None:
        intention = _FakeOneof("string_intention", string_intention="print('hi')")
        payload = _FakeOneof("intention", intention=intention)
        header = type("H", (), {"log_position": 42})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.log_position == 42
        assert result.type == "intention"
        assert result.content == "print('hi')"

    def test_vote_boolean_entry(self) -> None:
        abstract_vote = _FakeOneof("boolean_vote", boolean_vote=True)
        vote_info = _FakeOneof(
            "external_llm_vote_info",
            external_llm_vote_info=type(
                "V", (), {"reason": "safe", "model": "test-model"}
            )(),
        )
        vote = type(
            "Vote",
            (),
            {
                "intention_id": 10,
                "abstract_vote": abstract_vote,
                "info": vote_info,
            },
        )()
        payload = _FakeOneof("vote", vote=vote)
        header = type("H", (), {"log_position": 7})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "vote"
        assert result.boolean_vote is True
        assert result.intention_id == 10
        assert result.reason == "safe"
        assert result.model == "test-model"

    def test_vote_probability_entry(self) -> None:
        abstract_vote = _FakeOneof("probability_vote", probability_vote=0.85)
        vote_info = _FakeOneof(None)
        vote = type(
            "Vote",
            (),
            {
                "intention_id": 11,
                "abstract_vote": abstract_vote,
                "info": vote_info,
            },
        )()
        payload = _FakeOneof("vote", vote=vote)
        header = type("H", (), {"log_position": 8})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "vote"
        assert result.probability_vote == 0.85
        assert result.boolean_vote is None

    def test_commit_entry(self) -> None:
        commit = type("C", (), {"intention_id": 5, "reason": "approved"})()
        payload = _FakeOneof("commit", commit=commit)
        header = type("H", (), {"log_position": 20})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "commit"
        assert result.intention_id == 5
        assert result.reason == "approved"

    def test_abort_entry(self) -> None:
        abort = type("A", (), {"intention_id": 6, "reason": "rejected"})()
        payload = _FakeOneof("abort", abort=abort)
        header = type("H", (), {"log_position": 21})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "abort"
        assert result.intention_id == 6
        assert result.reason == "rejected"

    def test_inference_input_entry(self) -> None:
        ii = _FakeOneof(
            "string_inference_input",
            string_inference_input='{"messages": []}',
        )
        payload = _FakeOneof("inference_input", inference_input=ii)
        header = type("H", (), {"log_position": 30})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "inference_input"
        assert result.content == '{"messages": []}'

    def test_inference_output_entry(self) -> None:
        io = _FakeOneof(
            "string_inference_output",
            string_inference_output="Hello world",
        )
        payload = _FakeOneof("inference_output", inference_output=io)
        header = type("H", (), {"log_position": 31})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "inference_output"
        assert result.content == "Hello world"

    def test_action_output_entry(self) -> None:
        ao_inner = _FakeOneof(
            "string_action_output",
            string_action_output="result: 42",
        )
        ao = type(
            "AO",
            (),
            {
                "intention_id": 15,
                "WhichOneof": ao_inner.WhichOneof,
                "string_action_output": "result: 42",
            },
        )()
        payload = _FakeOneof("action_output", action_output=ao)
        header = type("H", (), {"log_position": 32})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "action_output"
        assert result.intention_id == 15
        assert result.content == "result: 42"

    def test_agent_input_entry(self) -> None:
        ai = _FakeOneof("string_agent_input", string_agent_input="user query")
        payload = _FakeOneof("agent_input", agent_input=ai)
        header = type("H", (), {"log_position": 33})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "agent_input"
        assert result.content == "user query"

    def test_agent_output_entry(self) -> None:
        ao = _FakeOneof("string_agent_output", string_agent_output="response text")
        payload = _FakeOneof("agent_output", agent_output=ao)
        header = type("H", (), {"log_position": 34})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "agent_output"
        assert result.content == "response text"

    def test_decider_policy_entry(self) -> None:
        decider_policy = "always_allow"
        payload = _FakeOneof("decider_policy", decider_policy=decider_policy)
        header = type("H", (), {"log_position": 1})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "decider_policy"
        assert result.policy == "always_allow"

    def test_voter_policy_entry(self) -> None:
        vp = type("VP", (), {"prompt_override": "custom prompt"})()
        payload = _FakeOneof("voter_policy", voter_policy=vp)
        header = type("H", (), {"log_position": 2})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "voter_policy"
        assert result.prompt_override == "custom prompt"

    def test_control_entry(self) -> None:
        ctrl = _FakeOneof("agent_input", agent_input="control message")
        payload = _FakeOneof("control", control=ctrl)
        header = type("H", (), {"log_position": 3})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "control"
        assert result.content == "control message"

    def test_unknown_payload(self) -> None:
        payload = _FakeOneof(None)
        header = type("H", (), {"log_position": 99})()
        entry = type("E", (), {"header": header, "payload": payload})()

        result = _proto_to_bus_entry(entry)
        assert result.type == "unknown"
        assert result.content is None


# ── BusService.entries() tests ────────────────────────────────────────


class TestBusServiceEntries:
    @pytest.mark.asyncio
    async def test_unknown_agent_raises(self, bus_service: BusService) -> None:
        with pytest.raises(KeyError, match="Agent not found"):
            async for _ in bus_service.entries("nonexistent-agent"):
                pass

    @pytest.mark.asyncio
    async def test_no_bus_configured_raises(
        self, bus_service: BusService, agent_without_bus: Agent
    ) -> None:
        with pytest.raises(ValueError, match="no agentbus configured"):
            async for _ in bus_service.entries(agent_without_bus.id):
                pass

    @pytest.mark.asyncio
    async def test_agent_with_bus_has_correct_config(
        self, registry: AgentRegistry, agent_with_bus: Agent
    ) -> None:
        """Verify agent's agentbus config is correctly stored."""
        agent = await registry.get(agent_with_bus.id)
        assert agent is not None
        assert agent.agentbus is not None
        assert agent.agentbus.url == "memory://9999"
        assert agent.agentbus.disable_safety is False


# ── AgentBusConfig tests ─────────────────────────────────────────────


class TestAgentBusConfig:
    def test_defaults(self) -> None:
        config = AgentBusConfig()
        assert config.url == "memory://"
        assert config.disable_safety is False

    def test_custom_url(self) -> None:
        config = AgentBusConfig(url="remote://localhost:9999", disable_safety=True)
        assert config.url == "remote://localhost:9999"
        assert config.disable_safety is True

    def test_memory_with_port(self) -> None:
        config = AgentBusConfig(url="memory://8095")
        assert config.url == "memory://8095"
