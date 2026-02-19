"""Unit tests for FleetTaskEnv."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def anyio_backend():
    return "asyncio"


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


@pytest.fixture
def mock_fleet_env_client():
    """Create a mock FleetEnvClient.from_fleet that returns mocks.

    Returns tools=None to avoid triggering asyncio.run() in __init__
    which conflicts with pytest-asyncio's event loop.
    """
    mock_orch = MagicMock()
    mock_orch._fleet_env = MagicMock()  # Fleet env handle for verifier

    with patch("envs.fleet_env.task_env.FleetEnvClient") as MockClient:
        # Return tools=None to skip the asyncio.run(list_tools()) call in __init__
        MockClient.from_fleet.return_value = (mock_orch, None)
        yield mock_orch, None


class TestFleetTaskEnvInit:
    """Tests for FleetTaskEnv initialization."""

    def test_init_with_api_key(self, sample_task_config, mock_fleet_env_client):
        """Should initialize with explicit API key."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test-api-key")
        assert env.api_key == "test-api-key"
        assert env.task_key == "test-task-001"
        assert env.prompt == "Search for flights from NYC to LA on January 15"
        assert env.modality == "tool_use"

    def test_init_from_env_var(
        self, sample_task_config, mock_fleet_env_client, monkeypatch
    ):
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

    def test_build_env_spec_with_version(
        self, sample_task_config, mock_fleet_env_client
    ):
        """Should build env_key:version spec."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        spec = env._build_env_spec()
        assert spec == "booking-com:v1.2.3"

    def test_build_env_spec_without_version(
        self, sample_task_config_no_version, mock_fleet_env_client
    ):
        """Should return just env_key when no version."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config_no_version, api_key="test")
        spec = env._build_env_spec()
        assert spec == "test-env"

    def test_get_data_key_with_data(self, sample_task_config, mock_fleet_env_client):
        """Should return data_key from config."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        assert env._get_data_key() == "consumer"
        assert env._get_data_version() == "v0.0.12"

    def test_get_data_key_without_data(
        self, sample_task_config_no_version, mock_fleet_env_client
    ):
        """Should return None when no data_key."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config_no_version, api_key="test")
        assert env._get_data_key() is None
        assert env._get_data_version() is None

    def test_build_env_spec_raises_without_env_key(self, mock_fleet_env_client):
        """Should raise when env_key is missing during init."""
        from envs.fleet_env.task_env import FleetTaskEnv

        task = {"task_key": "test", "prompt": "test"}
        # The error is raised during __init__ when _build_env_spec is called
        with pytest.raises(ValueError, match="missing env_key"):
            FleetTaskEnv(task, api_key="test")


class TestFleetTaskEnvVerifier:
    """Tests for verifier execution using Fleet SDK."""

    @pytest.mark.anyio
    async def test_compute_reward_returns_score_on_success(
        self, sample_task_config, mock_fleet_env_client
    ):
        """Should return verifier result score when Fleet SDK verifier succeeds."""
        from envs.fleet_env.task_env import FleetTaskEnv

        mock_orch, _ = mock_fleet_env_client
        env = FleetTaskEnv(sample_task_config, api_key="test")

        # Mock Fleet SDK Task.verify_detailed
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.result = 1.0

        with patch("fleet.tasks.Task") as MockTask:
            mock_task = MagicMock()
            mock_task.verify_detailed.return_value = mock_response
            MockTask.return_value = mock_task

            result = await env._compute_reward()
            assert result == 1.0
            mock_task.verify_detailed.assert_called_once_with(mock_orch._fleet_env)

    @pytest.mark.anyio
    async def test_compute_reward_returns_zero_on_failure(
        self, sample_task_config, mock_fleet_env_client
    ):
        """Should return 0.0 when Fleet SDK verifier fails."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")

        # Mock Fleet SDK Task.verify_detailed with failure
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.result = None

        with patch("fleet.tasks.Task") as MockTask:
            mock_task = MagicMock()
            mock_task.verify_detailed.return_value = mock_response
            MockTask.return_value = mock_task

            result = await env._compute_reward()
            assert result == 0.0

    @pytest.mark.anyio
    async def test_compute_reward_returns_zero_when_no_verifier(
        self, sample_task_config_no_version, mock_fleet_env_client
    ):
        """Should return 0.0 when no verifier code is present."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config_no_version, api_key="test")

        result = await env._compute_reward()
        assert result == 0.0

    @pytest.mark.anyio
    async def test_compute_reward_returns_zero_when_no_orch(
        self, sample_task_config, mock_fleet_env_client
    ):
        """Should return 0.0 when no orchestrator is available."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        env._orch = None

        result = await env._compute_reward()
        assert result == 0.0

    @pytest.mark.anyio
    async def test_compute_reward_returns_zero_when_no_fleet_env(
        self, sample_task_config, mock_fleet_env_client
    ):
        """Should return 0.0 when no Fleet env handle is available."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        env._orch._fleet_env = None  # No Fleet env handle

        result = await env._compute_reward()
        assert result == 0.0

    @pytest.mark.anyio
    async def test_compute_reward_handles_verifier_exception(
        self, sample_task_config, mock_fleet_env_client
    ):
        """Should return 0.0 when verifier raises an exception."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")

        with patch("fleet.tasks.Task") as MockTask:
            mock_task = MagicMock()
            mock_task.verify_detailed.side_effect = Exception("Verifier error")
            MockTask.return_value = mock_task

            result = await env._compute_reward()
            assert result == 0.0

    @pytest.mark.anyio
    async def test_compute_reward_handles_success_with_none_result(
        self, sample_task_config, mock_fleet_env_client
    ):
        """Should return 1.0 when verifier succeeds but returns None."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")

        mock_response = MagicMock()
        mock_response.success = True
        mock_response.result = None

        with patch("fleet.tasks.Task") as MockTask:
            mock_task = MagicMock()
            mock_task.verify_detailed.return_value = mock_response
            MockTask.return_value = mock_task

            result = await env._compute_reward()
            assert result == 1.0

    @pytest.mark.anyio
    async def test_compute_reward_supports_verifier_func_field(
        self, mock_fleet_env_client
    ):
        """Should support 'verifier_func' field name (Fleet SDK format)."""
        from envs.fleet_env.task_env import FleetTaskEnv

        # Task config using 'verifier_func' instead of 'verifier_code'
        task_config = {
            "task_key": "test-task-003",
            "prompt": "Test prompt",
            "env_key": "test-env",
            "verifier_func": "def verify(env): return 1.0",  # Fleet SDK field name
            "task_modality": "tool_use",
        }

        env = FleetTaskEnv(task_config, api_key="test")

        mock_response = MagicMock()
        mock_response.success = True
        mock_response.result = 1.0

        with patch("fleet.tasks.Task") as MockTask:
            mock_task = MagicMock()
            mock_task.verify_detailed.return_value = mock_response
            MockTask.return_value = mock_task

            result = await env._compute_reward()
            assert result == 1.0


class TestFleetTaskEnvFactories:
    """Tests for factory methods."""

    def test_make_fleet_task_env(self, sample_task_config, mock_fleet_env_client):
        """Should create FleetTaskEnv via factory function."""
        from envs.fleet_env.task_env import make_fleet_task_env

        env = make_fleet_task_env(sample_task_config, api_key="test")
        assert env.task_key == "test-task-001"


class TestFleetTaskEnvContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_closes_on_exit(
        self, sample_task_config, mock_fleet_env_client
    ):
        """Should close environment on context exit."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")

        with env:
            pass  # Context enters and exits

        # Environment should be closed
        assert env._orch is None
        assert env._tools is None
        assert env._done is True


class TestFleetTaskEnvProperties:
    """Tests for property accessors."""

    def test_task_key_property(self, sample_task_config, mock_fleet_env_client):
        """Should return task_key from config."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        assert env.task_key == "test-task-001"

    def test_task_key_default(self, mock_fleet_env_client):
        """Should return 'unknown' when task_key missing."""
        from envs.fleet_env.task_env import FleetTaskEnv

        task = {"prompt": "test", "env_key": "test-env"}
        env = FleetTaskEnv(task, api_key="test")
        assert env.task_key == "unknown"

    def test_prompt_property(self, sample_task_config, mock_fleet_env_client):
        """Should return prompt from config."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        assert env.prompt == "Search for flights from NYC to LA on January 15"

    def test_modality_property(self, sample_task_config, mock_fleet_env_client):
        """Should return task_modality from config."""
        from envs.fleet_env.task_env import FleetTaskEnv

        env = FleetTaskEnv(sample_task_config, api_key="test")
        assert env.modality == "tool_use"

    def test_modality_default(self, mock_fleet_env_client):
        """Should default to 'tool_use' when modality missing."""
        from envs.fleet_env.task_env import FleetTaskEnv

        task = {"task_key": "test", "prompt": "test", "env_key": "test-env"}
        env = FleetTaskEnv(task, api_key="test")
        assert env.modality == "tool_use"


class TestFleetTaskEnvComputerUseFiltering:
    """Tests for computer_use modality tool filtering."""

    @pytest.fixture
    def mock_fleet_env_with_tools(self):
        """Create mock FleetEnvClient that returns tools."""
        mock_orch = MagicMock()
        mock_tools = MagicMock()

        with patch("envs.fleet_env.task_env.FleetEnvClient") as MockClient:
            MockClient.from_fleet.return_value = (mock_orch, mock_tools)
            yield mock_orch, mock_tools

    @pytest.mark.anyio
    async def test_computer_use_filters_to_computer_tool(
        self, mock_fleet_env_with_tools
    ):
        """Should filter to only 'computer' tool for computer_use modality."""
        from envs.fleet_env.task_env import FleetTaskEnv

        mock_orch, mock_tools = mock_fleet_env_with_tools

        # Mock list_tools returning mixed tools (computer + API tools)
        async def mock_list_tools():
            return MagicMock(
                tools=[
                    {"name": "computer", "description": "Mouse/keyboard control"},
                    {"name": "search_issues", "description": "Search issues"},
                    {"name": "create_ticket", "description": "Create ticket"},
                ]
            )

        mock_tools.list_tools = mock_list_tools

        task_config = {
            "task_key": "test-task",
            "prompt": "Click on button",
            "env_key": "test-env",
            "task_modality": "computer_use",
        }

        env = FleetTaskEnv(task_config, api_key="test")
        obs = await env.reset_async()

        # Should only have computer tool
        assert len(env._tools_cache) == 1
        assert env._tools_cache[0]["name"] == "computer"

    @pytest.mark.anyio
    async def test_computer_use_clears_tools_when_no_computer_tool(
        self, mock_fleet_env_with_tools, caplog
    ):
        """Should clear tools and warn when no 'computer' tool for computer_use modality."""
        from envs.fleet_env.task_env import FleetTaskEnv
        import logging

        mock_orch, mock_tools = mock_fleet_env_with_tools

        # Mock list_tools returning only API tools (no computer tool)
        async def mock_list_tools():
            return MagicMock(
                tools=[
                    {"name": "search_issues", "description": "Search issues"},
                    {"name": "create_ticket", "description": "Create ticket"},
                ]
            )

        mock_tools.list_tools = mock_list_tools

        task_config = {
            "task_key": "sentry-task",
            "prompt": "Click on button",
            "env_key": "sentry",
            "task_modality": "computer_use",
        }

        env = FleetTaskEnv(task_config, api_key="test")

        with caplog.at_level(logging.WARNING):
            obs = await env.reset_async()

        # Should have empty tools (filtered out API tools)
        assert env._tools_cache == []

        # Should have logged warning
        assert "computer_use modality but no 'computer' tool found" in caplog.text

    @pytest.mark.anyio
    async def test_tool_use_excludes_computer_tool(self, mock_fleet_env_with_tools):
        """Should EXCLUDE computer tool for tool_use modality."""
        from envs.fleet_env.task_env import FleetTaskEnv

        mock_orch, mock_tools = mock_fleet_env_with_tools

        # Mock list_tools returning mixed tools (API tools + computer)
        async def mock_list_tools():
            return MagicMock(
                tools=[
                    {"name": "computer", "description": "Mouse/keyboard control"},
                    {"name": "search_issues", "description": "Search issues"},
                    {"name": "create_ticket", "description": "Create ticket"},
                ]
            )

        mock_tools.list_tools = mock_list_tools

        task_config = {
            "task_key": "test-task",
            "prompt": "Search for issues",
            "env_key": "test-env",
            "task_modality": "tool_use",  # tool_use, not computer_use
        }

        env = FleetTaskEnv(task_config, api_key="test")
        obs = await env.reset_async()

        # Should have only the 2 API tools (computer excluded)
        assert len(env._tools_cache) == 2
        tool_names = [t.get("name") for t in env._tools_cache]
        assert "computer" not in tool_names
        assert "search_issues" in tool_names
        assert "create_ticket" in tool_names

    @pytest.mark.anyio
    async def test_computer_use_filters_function_format(
        self, mock_fleet_env_with_tools
    ):
        """Should filter 'computer' tool from function format."""
        from envs.fleet_env.task_env import FleetTaskEnv

        mock_orch, mock_tools = mock_fleet_env_with_tools

        # Mock list_tools returning tools in OpenAI function format
        async def mock_list_tools():
            return MagicMock(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "computer", "description": "Control"},
                    },
                    {
                        "type": "function",
                        "function": {"name": "api_call", "description": "API"},
                    },
                ]
            )

        mock_tools.list_tools = mock_list_tools

        task_config = {
            "task_key": "test-task",
            "prompt": "Click button",
            "env_key": "test-env",
            "task_modality": "computer_use",
        }

        env = FleetTaskEnv(task_config, api_key="test")
        obs = await env.reset_async()

        # Should only have computer tool
        assert len(env._tools_cache) == 1
        assert env._tools_cache[0]["function"]["name"] == "computer"

    @pytest.mark.anyio
    async def test_tool_use_excludes_computer_function_format(
        self, mock_fleet_env_with_tools
    ):
        """Should exclude 'computer' tool from function format for tool_use."""
        from envs.fleet_env.task_env import FleetTaskEnv

        mock_orch, mock_tools = mock_fleet_env_with_tools

        # Mock list_tools returning tools in OpenAI function format
        async def mock_list_tools():
            return MagicMock(
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "computer", "description": "Control"},
                    },
                    {
                        "type": "function",
                        "function": {"name": "api_call", "description": "API"},
                    },
                ]
            )

        mock_tools.list_tools = mock_list_tools

        task_config = {
            "task_key": "test-task",
            "prompt": "Call API",
            "env_key": "test-env",
            "task_modality": "tool_use",
        }

        env = FleetTaskEnv(task_config, api_key="test")
        obs = await env.reset_async()

        # Should only have api_call tool (computer excluded)
        assert len(env._tools_cache) == 1
        assert env._tools_cache[0]["function"]["name"] == "api_call"
