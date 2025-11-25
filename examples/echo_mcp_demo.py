"""
Example: Using Echo Environment with MCP

This example demonstrates:
1. Connecting to echo_env server
2. Listing available tools via MCP
3. Calling tools using both step() API and direct tool methods
"""

import asyncio
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

    # List available tools
    print("2. Listing available tools...")
    tools = client.list_tools()
    for tool in tools:
        print(f"   - {tool['name']}: {tool['description']}")
    print()

    # Call echo_message tool using convenience method
    print("3. Calling echo_message tool...")
    result = client.echo_message("Hello from MCP!")
    print(f"   Result: {result}\n")

    # Check environment state
    print("4. Checking environment state...")
    state = client.state
    print(f"   Episode ID: {state.episode_id}")
    print(f"   Step count: {state.step_count}\n")

    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
