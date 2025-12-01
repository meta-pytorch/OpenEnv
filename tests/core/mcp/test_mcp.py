"""
Minimal tests for MCP infrastructure.

Tests the core MCP client/server integration and echo_env as reference.
"""

import pytest

from core.env_server.mcp_types import CallToolAction, ListToolsAction
from fastmcp import Client, FastMCP


@pytest.mark.asyncio
async def test_mcp_client_with_local_server():
    """Test FastMCP Client can list and call tools on a local FastMCP server."""
    # Create a simple MCP server
    mcp = FastMCP("test_server")

    @mcp.tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Create client connected to server (in-memory)
    client = Client(mcp)

    async with client:
        # Test list_tools
        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "add"
        assert "Add two numbers" in tools[0].description

        # Test call_tool
        result = await client.call_tool("add", {"a": 5, "b": 3})
        # FastMCP returns the raw result from the function
        assert result == 8


@pytest.mark.asyncio
async def test_echo_env_mcp_integration():
    """Test echo_env works with MCP actions (ListToolsAction, CallToolAction)."""
    from envs.echo_env.server.echo_environment import EchoEnvironment

    # Setup echo environment (MCPEnvironment handles MCP setup automatically)
    env = EchoEnvironment()

    # Test ListToolsAction
    list_action = ListToolsAction()
    obs = await env._handle_mcp_action(list_action)
    assert not obs.done
    assert hasattr(obs, "tools")
    assert len(obs.tools) == 1
    assert obs.tools[0].name == "echo_message"

    # Test CallToolAction
    call_action = CallToolAction(
        tool_name="echo_message", parameters={"message": "Hello MCP"}
    )
    obs = await env._handle_mcp_action(call_action)
    assert not obs.done
    assert hasattr(obs, "result")
    assert obs.error is None
    # Result is the dict returned by echo_message tool
    result = obs.result
    assert isinstance(result, dict)
    assert result["echoed_message"] == "Hello MCP"
