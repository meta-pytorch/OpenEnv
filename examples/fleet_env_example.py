"""
Example: Orchestrator + Agent loop using OpenEnv on Fleet.

Demonstrates the split architecture:
1. Orchestrator: Provisions environment, resets episodes (HTTP).
2. Agent: Lists tools, calls tools (MCP).

Prerequisites:
  pip install "openenv-core[fleet]"
  export FLEET_API_KEY="..."
  export FLEET_ENV_KEY="..."  # e.g. "browser-env" or your custom env
"""

import asyncio
import os
import random
import sys
from typing import Any, Dict, List, Sequence

# Ensure we can import from src/ if running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    # `openenv` installs top-level packages like `envs`, `core`, etc.
    # This example also prepends `src/` above so it works from a repo checkout.
    from envs.fleet_env import FleetEnvClient
except ImportError as e:
    raise ImportError(
        "Could not import `envs.fleet_env`. "
        "Run from the repo root, or install OpenEnv in editable mode: "
        "`python -m pip install -e '.[fleet]'`."
    ) from e

def get_openai_tool_param_enum(tool_def: Dict[str, Any], param_name: str) -> List[str]:
    """Extract an enum list for a parameter from an OpenAI 'tools' dict."""
    schema = tool_def.get("function", {}).get("parameters", {})
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        return []
    param_spec = props.get(param_name, {})
    if not isinstance(param_spec, dict):
        return []
    enum = param_spec.get("enum", [])
    return enum if isinstance(enum, list) else []

SAFE_COMPUTER_ACTION_PREFERENCE: Sequence[str] = ("screenshot", "wait", "cursor_position")


def pick_safe_computer_action(tool_def: Dict[str, Any]) -> str:
    """Pick a non-destructive default action for the Fleet 'computer' tool.

    Prefer safe actions like screenshot/wait, falling back to first enum.
    """
    actions = get_openai_tool_param_enum(tool_def, "action")
    if not actions:
        raise ValueError("Tool 'computer' has no available actions in schema.")

    action_set = set(actions)
    safe_available = [a for a in SAFE_COMPUTER_ACTION_PREFERENCE if a in action_set]
    if safe_available:
        return random.choice(safe_available)
    return actions[0]

def main():
    api_key = os.environ.get("FLEET_API_KEY")
    
    # 1. Get env_key from args or env var
    env_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FLEET_ENV_KEY")
    
    if not api_key or not env_key:
        print("Usage: python fleet_env_example.py <env_key>")
        print("   or: export FLEET_ENV_KEY=... && python fleet_env_example.py")
        raise ValueError("Please set FLEET_API_KEY and provide an env_key.")

    print(f"Provisioning Fleet environment: {env_key}...")
    
    # 1. Provision & Split Handles (Synchronous)
    # This must be run outside of an async loop because it manages its own loop.
    try:
        orch, tools = FleetEnvClient.from_fleet(
            api_key=api_key,
            env_key=env_key,
            ttl_seconds=600,  # 10 min TTL
        )
    except Exception as e:
        raise ValueError(f"Failed to provision environment: {e}")


    try:
        # Run the async agent loop
        asyncio.run(agent_loop(orch, tools))
    except BaseException as e:
        print(f"\n❌ Agent loop failed: {e}")
    finally:
        # 5. Cleanup (Synchronous)
        print("\nOrchestrator: Closing environment...")
        orch.close()
        print("Done.")


async def agent_loop(orch, tools):
    # 2. Orchestration: Start Episode (HTTP calls, sync method but we wrap or call directly)
    # orch.reset() is sync (requests), so it blocks the loop briefly. That's fine for this example.
    print("Orchestrator: Resetting environment...")
    obs = orch.reset()
    print(f"Reset complete. Initial observation keys: {list(obs.observation.metadata.keys())}")

    # 3. Agent: Discover Tools (Async)
    print("\nAgent: Discovering tools...")
    listed = await tools.list_tools()
    tool_defs = listed.tools
    print(f"Available tools ({len(tool_defs)}): {[t['function']['name'] for t in tool_defs]}")
    # Print the derived schema payloads (mirrors MCP Tool.inputSchema content, but OpenAI-shaped)
    print([t["function"]["parameters"] for t in tool_defs])

    if not tool_defs:
        print("No MCP tools available (all MCP endpoints may be down).")
        return

    # 4. Agent: Call a Tool
    target_tool_name = "computer"
    target_def = next((t for t in tool_defs if t["function"]["name"] == target_tool_name), None)

    if not target_def:
        print(f"Tool '{target_tool_name}' not found, picking first available.")
        target_def = tool_defs[0]
        target_tool_name = target_def["function"]["name"]

    print(f"\nTarget Tool: {target_tool_name}")
    # Inspect schema to construct params (in a real agent, the LLM does this)
    # schema = target_def["function"]["parameters"]
    # print(f"Schema: {json.dumps(schema, indent=2)}")

    params = {}
    if target_tool_name == "computer":
        # Choose a supported action from the schema (safe default).
        params = {"action": pick_safe_computer_action(target_def)}
    
    print(f"\nAgent: Calling tool '{target_tool_name}' with {params}...")
    result = await tools.call_tool(target_tool_name, params)

    
    # Result is typically a list of MCP content objects (TextContent/ImageContent)
    # We'll just print a summary.
    print("Agent: Tool execution result received.")
    print(f"{result=}")


if __name__ == "__main__":
    main()

