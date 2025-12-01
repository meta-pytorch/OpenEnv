"""
Example: Using Echo Environment with MCP

This example demonstrates:
1. Connecting to echo_env server
2. Listing available tools via MCP
3. Calling tools using the step() API
"""

import asyncio

try:
    from core.env_server.types import CallToolAction, ListToolsAction
except ImportError:
    from openenv_core.env_server.types import CallToolAction, ListToolsAction

from envs.echo_env import EchoEnv


async def main():
    # Connect to echo_env (assumes server is running on localhost:8000)
    # To start the server: uvicorn envs.echo_env.server.app:app
    client = EchoEnv(base_url="http://localhost:8000")

    print("=== Echo Environment MCP Demo ===\n")

    # Reset the environment
    print("1. Resetting environment...")
    result = client.reset()
    print(f"   Reset result: {result.observation.metadata}\n")

    # List available tools using step API
    print("2. Listing available tools...")
    list_action = ListToolsAction()
    list_result = client.step(list_action)
    if hasattr(list_result.observation, "tools"):
        for tool in list_result.observation.tools:
            print(f"   - {tool['name']}: {tool['description']}")
    print()

    # Call echo_message tool using step API
    print("3. Calling echo_message tool...")
    call_action = CallToolAction(tool_name="echo_message", parameters={"message": "Hello from MCP!"})
    call_result = client.step(call_action)
    print(f"   Result: {call_result.observation.result}\n")

    # Check environment state
    print("4. Checking environment state...")
    state = client.state
    print(f"   Episode ID: {state.episode_id}")
    print(f"   Step count: {state.step_count}\n")

    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
