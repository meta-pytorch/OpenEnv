import sys
import types

import pytest


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
        return _FakeResp(
            {"observation": {"metadata": {}}, "reward": 0.0, "done": False}
        )

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, None))
        return _FakeResp({"episode_id": "e1", "step_count": 0})


@pytest.fixture
def anyio_backend():
    # Avoid running the anyio test against trio (not installed in this repo env).
    return "asyncio"


@pytest.fixture
def fake_requests_session(monkeypatch):
    # Avoid importing real `requests` in this sandboxed environment (it may fail
    # while loading system CA bundles). core.http_env_client only needs Session.
    fake_requests = types.SimpleNamespace(Session=_FakeSession)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)


@pytest.fixture
def fake_fleet_module(monkeypatch):
    # Create a fake `fleet` module with Fleet.make returning an env with urls.
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


@pytest.mark.usefixtures("fake_requests_session", "fake_fleet_module")
def test_fleet_env_from_fleet_returns_orchestrator_and_tools():
    from envs.fleet_env import FleetEnvClient, FleetMCPTools

    orch, tools = FleetEnvClient.from_fleet(api_key="k", env_key="e")
    assert isinstance(orch, FleetEnvClient)
    assert isinstance(tools, FleetMCPTools)


@pytest.mark.usefixtures("fake_requests_session", "fake_fleet_module")
def test_fleet_env_reset_uses_http_manager_base_url():
    from envs.fleet_env import FleetEnvClient

    orch, _tools = FleetEnvClient.from_fleet(api_key="k", env_key="e")
    # reset() should hit {base}/reset
    _ = orch.reset()
    # access underlying fake session calls
    calls = orch._http.calls  # pylint: disable=protected-access
    assert calls[-1][0] == "POST"
    assert calls[-1][1].endswith("/reset")


@pytest.mark.usefixtures("fake_requests_session", "fake_fleet_module")
def test_fleet_env_step_rejects_tool_actions():
    from envs.fleet_env import FleetEnvClient, CallToolAction

    orch, _tools = FleetEnvClient.from_fleet(api_key="k", env_key="e")
    with pytest.raises(TypeError):
        orch.step(
            CallToolAction(tool_name="computer", parameters={"action": "screenshot"})
        )


@pytest.mark.anyio
async def test_agent_tools_list_and_call_routes(monkeypatch):
    from envs.fleet_env.mcp_tools import FleetMCPTools

    class _Tool:
        def __init__(self, name):
            self.name = name
            self.description = ""
            self.inputSchema = {"type": "object", "properties": {}, "required": []}

    class _FakeMCPClient:
        def __init__(self, url, api_key):
            self.url = url
            self.api_key = api_key
            self.list_calls = 0

        async def list_tools(self):
            self.list_calls += 1
            if self.url.endswith("api/v1/mcp"):
                return [_Tool("computer")]
            return [_Tool("search_issues")]

        async def call_tool(self, name, args):
            return {"url": self.url, "name": name, "args": args}

        def has_tool(self, name, tools_list=None):
            return any(t.name == name for t in (tools_list or []))

    monkeypatch.setattr("envs.fleet_env.mcp_tools.FleetMCPClient", _FakeMCPClient)

    tools = FleetMCPTools(
        api_key="k", mcp_urls=("https://x/api/v1/mcp", "https://x/mcp")
    )
    listed = await tools.list_tools()
    assert sorted([t["function"]["name"] for t in listed.tools]) == [
        "computer",
        "search_issues",
    ]

    res = await tools.call_tool("computer", {"action": "screenshot"})
    assert res["url"].endswith("api/v1/mcp")


class TestFleetMCPClientExtractToolResult:
    """Tests for FleetMCPClient._extract_tool_result()."""

    def test_extract_single_text_content(self):
        """Should extract text from single TextContent."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        # Mock CallToolResult with TextContent
        class _TextContent:
            type = "text"
            text = "file1.txt\nfile2.txt"

        class _Result:
            content = [_TextContent()]
            isError = False
            structuredContent = None

        result = client._extract_tool_result(_Result())
        assert result == "file1.txt\nfile2.txt"

    def test_extract_json_text_content(self):
        """Should parse JSON from text content."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        class _TextContent:
            type = "text"
            text = '{"status": "success", "count": 42}'

        class _Result:
            content = [_TextContent()]
            isError = False
            structuredContent = None

        result = client._extract_tool_result(_Result())
        assert result == {"status": "success", "count": 42}

    def test_extract_multiple_text_contents(self):
        """Should return list when multiple text contents."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        class _TextContent1:
            type = "text"
            text = "first"

        class _TextContent2:
            type = "text"
            text = "second"

        class _Result:
            content = [_TextContent1(), _TextContent2()]
            isError = False
            structuredContent = None

        result = client._extract_tool_result(_Result())
        assert result == ["first", "second"]

    def test_extract_error_result(self):
        """Should return error dict when isError=True."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        class _TextContent:
            type = "text"
            text = "Tool failed: permission denied"

        class _Result:
            content = [_TextContent()]
            isError = True
            structuredContent = None

        result = client._extract_tool_result(_Result())
        assert result == {"error": "Tool failed: permission denied"}

    def test_extract_structured_content_fallback(self):
        """Should use structuredContent when no text content."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        class _Result:
            content = []
            isError = False
            structuredContent = {"data": [1, 2, 3]}

        result = client._extract_tool_result(_Result())
        assert result == {"data": [1, 2, 3]}

    def test_extract_empty_result(self):
        """Should return string repr for empty result."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        class _Result:
            content = []
            isError = False
            structuredContent = None

            def __str__(self):
                return "EmptyResult()"

        result = client._extract_tool_result(_Result())
        assert result == "EmptyResult()"

    def test_extract_image_content(self):
        """Should extract ImageContent as OpenAI-compatible format."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        # Mock MCP ImageContent
        class _ImageContent:
            type = "image"
            data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            mimeType = "image/png"

        class _Result:
            content = [_ImageContent()]
            isError = False
            structuredContent = None

        result = client._extract_tool_result(_Result())

        # Should return list with single image_url item
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert "image_url" in result[0]
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_extract_mixed_text_and_image_content(self):
        """Should extract mixed text and image content."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        class _TextContent:
            type = "text"
            text = "Screenshot captured"

        class _ImageContent:
            type = "image"
            data = "base64imagedata"
            mimeType = "image/jpeg"

        class _Result:
            content = [_TextContent(), _ImageContent()]
            isError = False
            structuredContent = None

        result = client._extract_tool_result(_Result())

        # Should return list with both items
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Screenshot captured"
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "data:image/jpeg;base64,base64imagedata"

    def test_extract_image_default_mimetype(self):
        """Should default to image/png when mimeType is missing."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        class _ImageContent:
            type = "image"
            data = "somebase64data"
            mimeType = None  # Missing mimeType

        class _Result:
            content = [_ImageContent()]
            isError = False
            structuredContent = None

        result = client._extract_tool_result(_Result())

        assert isinstance(result, list)
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_extract_base64_image_json_format(self):
        """Should convert Fleet MCP's base64_image JSON format to OpenAI format."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        # Fleet MCP returns screenshot as JSON text with base64_image key
        class _TextContent:
            type = "text"
            text = '{"base64_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."}'

        class _Result:
            content = [_TextContent()]
            isError = False
            structuredContent = None

        result = client._extract_tool_result(_Result())

        # Should be converted to OpenAI-compatible format
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert (
            result[0]["image_url"]["url"] == "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        )

    def test_extract_base64_image_preserves_other_json(self):
        """Should preserve normal JSON responses that don't have base64_image."""
        from envs.fleet_env.fleet_mcp_client import FleetMCPClient

        client = FleetMCPClient(url="http://test", api_key="test")

        # Normal JSON response without base64_image
        class _TextContent:
            type = "text"
            text = '{"status": "success", "data": [1, 2, 3]}'

        class _Result:
            content = [_TextContent()]
            isError = False
            structuredContent = None

        result = client._extract_tool_result(_Result())

        # Should return parsed dict as-is
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["data"] == [1, 2, 3]


class TestFleetTaskEnvInitFetchesTools:
    """Tests for FleetTaskEnv fetching tools during __init__()."""

    def test_init_fetches_tools(self, monkeypatch):
        """__init__ should create env and fetch tools."""
        from unittest.mock import MagicMock

        mock_orch = MagicMock()
        mock_tools = MagicMock()

        # Create a proper coroutine for list_tools
        async def mock_list_tools():
            return MagicMock(tools=[{"type": "function", "function": {"name": "bash"}}])

        mock_tools.list_tools = mock_list_tools

        # Monkeypatch BEFORE importing/creating FleetTaskEnv
        monkeypatch.setattr(
            "envs.fleet_env.task_env.FleetEnvClient.from_fleet",
            lambda **kwargs: (mock_orch, mock_tools),
        )

        from envs.fleet_env.task_env import FleetTaskEnv

        task_config = {
            "task_key": "test-task",
            "prompt": "Test prompt",
            "env_key": "test-env",
            "task_modality": "tool_use",
        }

        # Tools should be fetched during __init__
        env = FleetTaskEnv(task_config, api_key="test-key")

        # Verify tools were cached
        assert env._tools_cache is not None
        assert len(env._tools_cache) == 1
        assert env._tools_cache[0]["function"]["name"] == "bash"

    def test_reset_returns_cached_tools(self, monkeypatch):
        """reset() should return cached tools from __init__."""
        from unittest.mock import MagicMock

        mock_orch = MagicMock()
        mock_tools = MagicMock()
        list_tools_call_count = 0

        # Create a proper coroutine for list_tools that tracks calls
        async def mock_list_tools():
            nonlocal list_tools_call_count
            list_tools_call_count += 1
            return MagicMock(
                tools=[{"type": "function", "function": {"name": "search"}}]
            )

        mock_tools.list_tools = mock_list_tools

        monkeypatch.setattr(
            "envs.fleet_env.task_env.FleetEnvClient.from_fleet",
            lambda **kwargs: (mock_orch, mock_tools),
        )

        from envs.fleet_env.task_env import FleetTaskEnv

        task_config = {
            "task_key": "test-task",
            "prompt": "Test prompt",
            "env_key": "test-env",
            "task_modality": "tool_use",
        }

        env = FleetTaskEnv(task_config, api_key="test-key")

        # reset should return cached tools (no new fetch)
        obs = env.reset()

        assert "tools" in obs
        assert len(obs["tools"]) == 1
        assert obs["tools"][0]["function"]["name"] == "search"

        # Verify list_tools was only called once (during __init__)
        assert list_tools_call_count == 1

    def test_reset_sync_returns_cached_tools(self, monkeypatch):
        """Sync reset() should return cached tools."""
        from unittest.mock import MagicMock

        mock_orch = MagicMock()
        mock_tools = MagicMock()

        # Create a proper coroutine for list_tools
        async def mock_list_tools():
            return MagicMock(
                tools=[{"type": "function", "function": {"name": "computer"}}]
            )

        mock_tools.list_tools = mock_list_tools

        monkeypatch.setattr(
            "envs.fleet_env.task_env.FleetEnvClient.from_fleet",
            lambda **kwargs: (mock_orch, mock_tools),
        )

        from envs.fleet_env.task_env import FleetTaskEnv

        task_config = {
            "task_key": "test-task",
            "prompt": "Test prompt",
            "env_key": "test-env",
            "task_modality": "tool_use",
        }

        env = FleetTaskEnv(task_config, api_key="test-key")

        # Sync reset should return cached tools
        obs = env.reset()

        assert "tools" in obs
        assert len(obs["tools"]) == 1
        assert obs["tools"][0]["function"]["name"] == "computer"
