#!/usr/bin/env python3
"""Quick test script to verify MCP integration works."""

import asyncio
import sys
sys.path.insert(0, 'src')

from envs.echo_env.server.echo_environment import EchoEnvironment
from core.env_server.mcp_types import ListToolsAction, CallToolAction


async def main():
    print("=" * 60)
    print("Testing MCP Integration")
    print("=" * 60)

    # Create echo environment (MCPEnvironment handles MCP setup automatically)
    print("\n1. Creating Echo Environment...")
    env = EchoEnvironment()

    # Test list tools
    print("\n2. Testing ListToolsAction...")
    list_action = ListToolsAction()
    obs = await env._handle_mcp_action(list_action)
    print(f"   - Done: {obs.done}")
    print(f"   - Has 'tools' attribute: {hasattr(obs, 'tools')}")
    if hasattr(obs, "tools"):
        print(f"   - Number of tools: {len(obs.tools)}")
        print(f"   - Tool names: {[t.name for t in obs.tools]}")
    else:
        print("   - ERROR: No 'tools' attribute!")
        return False

    # Test call tool
    print("\n3. Testing CallToolAction...")
    call_action = CallToolAction(
        tool_name="echo_message",
        parameters={"message": "Hello MCP!"}
    )
    obs = await env._handle_mcp_action(call_action)
    print(f"   - Done: {obs.done}")
    print(f"   - Has 'result' attribute: {hasattr(obs, 'result')}")
    print(f"   - Error: {obs.error}")
    if hasattr(obs, "result") and obs.result is not None:
        result = obs.result
        print(f"   - Result type: {type(result)}")
        print(f"   - Result: {result}")
    else:
        print("   - ERROR: No 'result' attribute or result is None!")
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
