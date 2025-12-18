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
        return _FakeResp({"observation": {"metadata": {}}, "reward": 0.0, "done": False})

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
        orch.step(CallToolAction(tool_name="computer", parameters={"action": "screenshot"}))


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

    tools = FleetMCPTools(api_key="k", mcp_urls=("https://x/api/v1/mcp", "https://x/mcp"))
    listed = await tools.list_tools()
    assert sorted([t["function"]["name"] for t in listed.tools]) == ["computer", "search_issues"]

    res = await tools.call_tool("computer", {"action": "screenshot"})
    assert res["url"].endswith("api/v1/mcp")


