"""
Complete LLM agent example for AWM environment.

Usage:
    # Terminal 1: Start the server
    PYTHONPATH=src:envs uv run uvicorn \
        envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899

    # Terminal 2: Run the agent (set LLM credentials first, you can use any openai compatible LLM)
    export ENDPOINT_URL="https://YOUR_ENDPOINT_URL/v1"
    export OPENAI_API_KEY="your-api-key"
    export AWM_EXAMPLE_AGENT_MODEL="gpt-5"
    PYTHONPATH=src:envs uv run python envs/agent_world_model_env/example_usage.py

    # Optional: set LLM credentials for SQL verifier mode
    export OPENENV_AWM_LLM_BASE_URL="https://..."
    export OPENENV_AWM_LLM_API_KEY="..."
    export OPENENV_AWM_LLM_MODEL="gpt-5"
"""

import asyncio
import json
import os
import re

from openai import AsyncOpenAI
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv, AWMObservation


SYSTEM_PROMPT = """\
You are at a MCP environment. You need to call MCP tools to assist with the user query. \
At each step, you can only call one function. You have already logged in, and your user id is 1 if required.

You are provided with TWO functions:

1. list_tools
   - Description: List all available MCP tools for the current environment.
   - Arguments: None

2. call_tool
   - Description: Call a MCP environment-specific tool
   - Arguments:
       - tool_name: str, required
       - arguments: str, required, valid JSON string

For each function call, return a json object within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Example:
<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}}
</tool_call>

You should call list_tools first to discover available tools, then use call_tool to interact. \
When you have enough information to answer, output the answer directly without any tool_call tags."""


def parse_tool_call(content: str) -> dict | None:
    """Extract the first <tool_call> block from LLM output."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or "name" not in data:
        return None
    return data


def format_tools(tools) -> str:
    """Format Tool objects into a readable string for the LLM."""
    lines = [f"Available MCP Tools ({len(tools)} tools):", "=" * 60]
    for i, t in enumerate(tools, 1):
        lines.append(f"{i}. {t.name}")
        lines.append(f"   Description: {t.description}")
        props = t.input_schema.get("properties", {})
        required = t.input_schema.get("required", [])
        if props:
            lines.append("   Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                lines.append(
                    f"     - {pname}: {pinfo.get('type', 'any')}{req} — {pinfo.get('description', '')}"
                )
        else:
            lines.append("   Parameters: None")
        lines.append("")
    return "\n".join(lines)


async def main():
    async with AWMEnv(base_url="http://localhost:8899") as env:
        # =====================================================================
        # 1. List all scenarios (1,000 scenarios x 10 tasks each)
        # =====================================================================
        result: StepResult[AWMObservation] = await env.step(
            CallToolAction(tool_name="__list_scenarios__", arguments={})
        )
        print(
            "total scenarios:",
            result.observation.total,
            len(result.observation.scenarios),
        )
        assert len(result.observation.scenarios) == result.observation.total == 1000, (
            "total scenarios should be 1000"
        )
        assert all(len(s["tasks"]) == 10 for s in result.observation.scenarios), (
            "each scenario should have 10 tasks"
        )
        print("=" * 100)
        for scenario in result.observation.scenarios[:3]:
            print(
                "scenario:",
                scenario["name"],
                "task num",
                len(scenario["tasks"]),
                "sample task:",
                scenario["tasks"][0],
            )
        print("=" * 100)

        # =====================================================================
        # 2. Reset to a specific scenario and task
        # =====================================================================
        # Reset returns verifier support info (has_verifier: {sql: bool, code: bool} or None)
        # Pass LLM credentials for sql verifier mode (or set via OPENENV_AWM_LLM_* env vars)
        result: StepResult[AWMObservation] = await env.reset(
            scenario="e_commerce_33",
            task_idx=0,
            llm_base_url=os.environ.get("OPENENV_AWM_LLM_BASE_URL"),
            llm_api_key=os.environ.get("OPENENV_AWM_LLM_API_KEY"),
            llm_model=os.environ.get("OPENENV_AWM_LLM_MODEL"),
        )
        task_description = result.observation.task
        print(
            "reset result:",
            f"scenario: {result.observation.scenario}, "
            f"task: {task_description}, "
            f"has_verifier: {result.observation.has_verifier}, "
            f"total tools: {result.observation.num_tools}",
        )
        print("=" * 100)

        # =====================================================================
        # 3. List tools for this scenario
        # =====================================================================
        result: StepResult[AWMObservation] = await env.step(ListToolsAction())
        print("list tools results", f"total tools: {len(result.observation.tools)}")
        for tool in result.observation.tools[:3]:
            print(f"Tool: {tool.name}, Description: {tool.description}")
            print(f"Input Schema: {tool.input_schema}")
            print("=" * 100)

        # pre-format tool list text for agent loop
        tools_text = format_tools(result.observation.tools)

        # =====================================================================
        # 4. Agent loop — LLM iteratively calls tools
        # =====================================================================
        # Set LLM credentials: export ENDPOINT_URL and OPENAI_API_KEY
        print("=" * 100)
        print("Agent loop starts")
        print("=" * 100)

        MAX_ITERATIONS = 5
        TEMPERATURE = 1.0
        MAX_TOKENS = 2048
        model = os.environ.get("AWM_EXAMPLE_AGENT_MODEL", "gpt-5")

        llm = AsyncOpenAI(
            base_url=os.environ["ENDPOINT_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
        )

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_description},
        ]

        for step in range(1, MAX_ITERATIONS + 1):
            response = await llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
            )
            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": content})

            tc = parse_tool_call(content)
            if not tc:
                print(f"\n[Step {step}] Final answer:\n{content}")
                break

            name = tc["name"]
            arguments = tc.get("arguments") or {}
            print(
                f"[Step {step}] Tool call: {name} "
                f"{json.dumps(arguments, ensure_ascii=False)[:200]}"
            )

            if name == "list_tools":
                result = await env.step(ListToolsAction())
                tool_response = format_tools(result.observation.tools)
            elif name == "call_tool":
                tool_name = arguments.get("tool_name", "")
                inner_args = arguments.get("arguments", "{}")
                if isinstance(inner_args, str):
                    try:
                        inner_args = json.loads(inner_args)
                    except json.JSONDecodeError:
                        inner_args = {}
                if not isinstance(inner_args, dict):
                    inner_args = {}

                result = await env.step(
                    CallToolAction(tool_name=tool_name, arguments=inner_args)
                )
                obs = result.observation
                if hasattr(obs, "tool_result") and obs.tool_result is not None:
                    tool_response = (
                        json.dumps(obs.tool_result, ensure_ascii=False)
                        if not isinstance(obs.tool_result, str)
                        else obs.tool_result
                    )
                elif hasattr(obs, "error") and obs.error:
                    tool_response = f"Error: {obs.error}"
                else:
                    tool_response = json.dumps(obs.model_dump(), ensure_ascii=False)
            else:
                tool_response = (
                    f"Error: Unknown tool '{name}'. Use 'list_tools' or 'call_tool'."
                )

            print(f"  -> Response: {tool_response[:200]}...Reward: {result.reward}")
            messages.append(
                {"role": "user", "content": f"Tool response:\n{tool_response}"}
            )
        else:
            print(f"Max iterations ({MAX_ITERATIONS}) reached.")

        # =====================================================================
        # 5. Verification — call verify with different modes
        # =====================================================================
        print("=" * 100)
        result: StepResult[AWMObservation] = await env.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": content},
            )
        )
        print("code verifier result:", result.observation.verify_result)
        print("reward_type:", result.observation.reward_type, "reward:", result.reward)
        print("=" * 100)

        result: StepResult[AWMObservation] = await env.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "sql"},
            )
        )
        print("sql verifier result:", result.observation.verify_result)
        print("reward_type:", result.observation.reward_type, "reward:", result.reward)
        print("=" * 100)

        # =====================================================================
        # 6. End episode — keep_session=True preserves all session artifacts
        #    (trajectory.json, DBs, server.py, server.log)
        # =====================================================================
        result: StepResult[AWMObservation] = await env.step(
            CallToolAction(tool_name="done", arguments={"keep_session": True})
        )
        print("episode done:", result.done)
        print("trajectory_path:", result.observation.trajectory_path)
        print("session_dir:", result.observation.session_dir)


if __name__ == "__main__":
    # Start the server first:
    #   PYTHONPATH=src:envs uv run uvicorn \
    #       envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899
    #
    # For SQL verifier mode, export:
    #   OPENENV_AWM_LLM_BASE_URL, OPENENV_AWM_LLM_API_KEY, OPENENV_AWM_LLM_MODEL

    asyncio.run(main())
