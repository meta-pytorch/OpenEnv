"""Unit tests for FleetTaskEnv."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import types


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json))
        return _FakeResp({"observation": {"metadata": {}}, "reward": 0.0, "done": False})

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, None))
        return _FakeResp({"episode_id": "e1", "step_count": 0})


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fake_requests_session(monkeypatch):
    fake_requests = types.SimpleNamespace(Session=_FakeSession)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)


@pytest.fixture
def fake_fleet_module(monkeypatch):
    """Create a fake fleet module with Fleet.make returning an env with urls."""

    class _Urls:
        def __init__(self):
            self.root = "https://example/"

            class _Mgr:
                api = "https://example/api/v1/env"

            self.manager = _Mgr()

    class _Env:
        def __init__(self):
            self.urls = _Urls()
            self.closed = False

        def close(self):
            self.closed = True

    class _Fleet:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def make(self, **kwargs):
            return _Env()

    mod = types.SimpleNamespace(Fleet=_Fleet)
    monkeypatch.setitem(sys.modules, "fleet", mod)


@pytest.fixture
def sample_task_config():
    """Sample task configuration for testing."""
    return {
        "task_key": "test-task-001",
        "prompt": "Search for flights from NYC to LA on January 15",
        "env_key": "booking-com",
        "env_version": "v1.2.3",
        "data_key": "consumer",
        "data_version": "v0.0.12",
        "verifier_code": "async def verify(env): return True",
        "task_modality": "tool_use",
        "tool_use_workflow": [{"tool": "search"}],
    }


@pytest.fixture
def sample_task_config_no_version():
    """Task config without version info."""
    return {
        "task_key": "test-task-002",
        "prompt": "Test prompt",
        "env_key": "test-env",
        "task_modality": "tool_use",
    }


class TestFleetTaskEnvInit:
    """Tests for FleetTaskEnv initialization."""

    def test_init_with_api_key(self, sample_task_config):
        """Should initialize with explicit API key."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test-api-key")
        assert env.api_key == "test-api-key"
        assert env.task_key == "test-task-001"
        assert env.prompt == "Search for flights from NYC to LA on January 15"
        assert env.modality == "tool_use"

    def test_init_from_env_var(self, sample_task_config, monkeypatch):
        """Should use FLEET_API_KEY env var if no api_key provided."""
        from envs.fleet_env.task_env import FleetTaskEnv

        monkeypatch.setenv("FLEET_API_KEY", "env-api-key")
        env = FleetTaskEnv(sample_task_config)
        assert env.api_key == "env-api-key"

    def test_init_raises_without_api_key(self, sample_task_config, monkeypatch):
        """Should raise if no API key available."""
        from envs.fleet_env.task_env import FleetTaskEnv

        monkeypatch.delenv("FLEET_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Fleet API key required"):
            FleetTaskEnv(sample_task_config)


class TestFleetTaskEnvSpecs:
    """Tests for env/data spec building."""

    def test_build_env_spec_with_version(self, sample_task_config):
        """Should build env_key:version spec."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        spec = env._build_env_spec()
        assert spec == "booking-com:v1.2.3"

    def test_build_env_spec_without_version(self, sample_task_config_no_version):
        """Should return just env_key when no version."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config_no_version, api_key="test")
        spec = env._build_env_spec()
        assert spec == "test-env"

    def test_build_data_spec_with_version(self, sample_task_config):
        """Should build data_key:version spec."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        spec = env._build_data_spec()
        assert spec == "consumer:v0.0.12"

    def test_build_data_spec_without_data_key(self, sample_task_config_no_version):
        """Should return None when no data_key."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config_no_version, api_key="test")
        spec = env._build_data_spec()
        assert spec is None

    def test_build_env_spec_raises_without_env_key(self):
        """Should raise when env_key is missing."""
        from envs.fleet_env.task_env import FleetTaskEnv

        task = {"task_key": "test", "prompt": "test"}
        env = FleetTaskEnv(task, api_key="test")
        with pytest.raises(ValueError, match="missing env_key"):
            env._build_env_spec()


class TestFleetTaskEnvVerifier:
    """Tests for verifier execution."""

    @pytest.mark.anyio
    async def test_execute_verifier_local_returns_true(self, sample_task_config):
        """Should return True when verifier passes."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        env._orch = MagicMock()

        verifier_code = "async def verify(env): return True"
        result = await env._execute_verifier_local(verifier_code)
        assert result is True

    @pytest.mark.anyio
    async def test_execute_verifier_local_returns_false(self, sample_task_config):
        """Should return False when verifier fails."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        env._orch = MagicMock()

        verifier_code = "async def verify(env): return False"
        result = await env._execute_verifier_local(verifier_code)
        assert result is False

    @pytest.mark.anyio
    async def test_execute_verifier_local_handles_numeric_result(self, sample_task_config):
        """Should handle numeric verifier results."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        env._orch = MagicMock()

        # Positive number = pass
        verifier_code = "async def verify(env): return 1.0"
        result = await env._execute_verifier_local(verifier_code)
        assert result is True

        # Zero = fail
        verifier_code = "async def verify(env): return 0.0"
        result = await env._execute_verifier_local(verifier_code)
        assert result is False

    @pytest.mark.anyio
    async def test_execute_verifier_local_handles_dict_result(self, sample_task_config):
        """Should handle dict verifier results."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        env._orch = MagicMock()

        # success=True
        verifier_code = "async def verify(env): return {'success': True}"
        result = await env._execute_verifier_local(verifier_code)
        assert result is True

        # score > 0
        verifier_code = "async def verify(env): return {'score': 1.0}"
        result = await env._execute_verifier_local(verifier_code)
        assert result is True

        # score = 0
        verifier_code = "async def verify(env): return {'score': 0}"
        result = await env._execute_verifier_local(verifier_code)
        assert result is False

    @pytest.mark.anyio
    async def test_execute_verifier_local_raises_on_missing_function(self, sample_task_config):
        """Should raise when verify function not defined."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        env._orch = MagicMock()

        verifier_code = "x = 1"  # No verify function
        with pytest.raises(ValueError, match="must define a 'verify' function"):
            await env._execute_verifier_local(verifier_code)


class TestFleetTaskEnvFactories:
    """Tests for factory methods."""

    def test_make_fleet_task_env(self, sample_task_config):
        """Should create FleetTaskEnv via factory function."""
        from envs.fleet_env.task_env import make_fleet_task_env

        env = make_fleet_task_env(sample_task_config, api_key="test")
        assert isinstance(env, object)  # Can't import FleetTaskEnv here
        assert env.task_key == "test-task-001"


class TestFleetTaskEnvContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_closes_on_exit(self, sample_task_config):
        """Should close environment on context exit."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        env._orch = MagicMock()
        env._tools = MagicMock()

        with env:
            pass  # Context enters and exits

        # Environment should be closed
        assert env._orch is None
        assert env._tools is None
        assert env._done is True


class TestFleetTaskEnvProperties:
    """Tests for property accessors."""

    def test_task_key_property(self, sample_task_config):
        """Should return task_key from config."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        assert env.task_key == "test-task-001"

    def test_task_key_default(self):
        """Should return 'unknown' when task_key missing."""
        from envs.fleet_env.task_env import FleetTaskEnv

        task = {"prompt": "test", "env_key": "test-env"}
        env = FleetTaskEnv(task, api_key="test")
        assert env.task_key == "unknown"

    def test_prompt_property(self, sample_task_config):
        """Should return prompt from config."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        assert env.prompt == "Search for flights from NYC to LA on January 15"

    def test_modality_property(self, sample_task_config):
        """Should return task_modality from config."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        assert env.modality == "tool_use"

    def test_modality_default(self):
        """Should default to 'tool_use' when modality missing."""
        from envs.fleet_env.task_env import FleetTaskEnv

        task = {"task_key": "test", "prompt": "test", "env_key": "test-env"}
        env = FleetTaskEnv(task, api_key="test")
        assert env.modality == "tool_use"
