import argparse
import asyncio
import json
import os
import re

from agent_world_model_env import AWMEnv, AWMObservation
from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction
from openenv.core.client_types import StepResult
from openai import AsyncAzureOpenAI


AGENT_SYSTEM_PROMPT = """\
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

IMPORTANT RULES:
- Always call list_tools first to discover available tools.
- Do NOT use "playbook" shortcut tools. Always use the granular/primitive tools (e.g. create_connector, create_workflow, create_workflow_action_step, update_workflow, etc.) to ensure every parameter matches the task requirements exactly.
- BEFORE creating any resource, first check if it already exists (e.g. list action steps for a workflow before creating one, get a workflow by name before creating it). The environment may have pre-existing data. If a resource already exists, use the update/patch endpoint instead of create.
- If a tool call returns an error (e.g. UNIQUE constraint, 500, validation error), read the error carefully. It often means the resource already exists — try updating it instead, or adjust your approach.
- After performing actions, verify the result by reading back the created/updated resource to confirm it matches the task requirements.
- Make sure all arguments are passed as proper JSON with correct types.
- When you have enough information to answer, output the answer directly without any tool_call tags."""


EXPERT_SYSTEM_PROMPT = """\
You are an expert advisor for an AI agent that interacts with MCP tool-use environments. \
You have access to the full output log of a previous baseline run of the same scenario. \
Your job is to analyze what happened for a specific task in the baseline and provide \
concise, actionable advice to help the agent succeed.

BASELINE RUN LOG:
--- START LOG ---
{baseline_log}
--- END LOG ---

When asked about a specific task:
1. Find the task in the baseline log (look for "# Scenario: ..., Task: N").
2. Determine if it PASSED (reward=1.0) or FAILED.
3. If it PASSED: briefly summarize the successful strategy (which tools were called, in what order).
4. If it FAILED: diagnose the root cause (e.g. UNIQUE constraint error, wrong tool used, missing step, etc.) and suggest a concrete fix.
5. Note any pre-existing data the agent should check for before creating resources.
6. Keep your advice under 200 words. Be specific about tool names and argument values."""


def parse_tool_call(content: str) -> dict | None:
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


class ExpertAdvisor:
    def __init__(self, baseline_log_path: str, llm: AsyncAzureOpenAI, model: str):
        with open(baseline_log_path, "r", encoding="utf-8") as f:
            self._baseline_log = f.read()
        self._llm = llm
        self._model = model
        self._system_prompt = EXPERT_SYSTEM_PROMPT.format(
            baseline_log=self._baseline_log
        )

    async def advise(self, scenario: str, task_idx: int, task_description: str) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    f"I am about to run Scenario: {scenario}, Task: {task_idx}.\n"
                    f"Task description: {task_description}\n\n"
                    "What happened in the baseline run for this task? "
                    "Give me concise, actionable advice."
                ),
            },
        ]
        response = await self._llm.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=512,
        )
        return response.choices[0].message.content or ""


async def run_task(
    scenario: str,
    task_idx: int,
    expert: ExpertAdvisor,
    agent_llm: AsyncAzureOpenAI,
    agent_model: str,
):
    print(f"\n{'#'*80}")
    print(f"# Scenario: {scenario}, Task: {task_idx}")
    print(f"{'#'*80}")

    async with AWMEnv(base_url="http://localhost:8899") as env:
        result = await env.reset(scenario=scenario, task_idx=task_idx)
        task_description = result.observation.task
        print(f"Task: {task_description}")
        print(f"Has verifier: {result.observation.has_verifier}, Tools: {result.observation.num_tools}")
        print("-" * 80)

        # Consult the expert before starting
        print("  [Expert] Consulting expert advisor...")
        advice = await expert.advise(scenario, task_idx, task_description)
        print(f"  [Expert] Advice: {advice[:300]}...")
        print("-" * 80)

        result = await env.step(ListToolsAction())
        tools_text = format_tools(result.observation.tools)

        MAX_ITERATIONS = 15

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"[Expert Advisor]: Based on previous runs of this scenario, "
                    f"here is advice for this task:\n{advice}\n\n"
                    f"Now here is the task:\n{task_description}"
                ),
            },
        ]

        content = ""
        for step in range(1, MAX_ITERATIONS + 1):
            response = await agent_llm.chat.completions.create(
                model=agent_model,
                messages=messages,
                temperature=1.0,
                max_completion_tokens=2048,
            )
            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": content})

            tc = parse_tool_call(content)
            if not tc:
                print(f"  [Step {step}] Final answer: {content[:200]}")
                break

            name = tc["name"]
            arguments = tc.get("arguments") or {}
            print(f"  [Step {step}] {name} {json.dumps(arguments, ensure_ascii=False)[:150]}")

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
                tool_response = f"Error: Unknown tool '{name}'."

            print(f"    -> {tool_response[:150]}...")
            messages.append({"role": "user", "content": f"Tool response:\n{tool_response}"})
        else:
            print(f"  Max iterations ({MAX_ITERATIONS}) reached.")

        result = await env.step(
            CallToolAction(
                tool_name="verify",
                arguments={"verifier_mode": "code", "final_answer": content},
            )
        )
        reward = result.reward
        reward_type = result.observation.reward_type
        print(f"  RESULT: reward_type={reward_type}, reward={reward}")

        await env.step(CallToolAction(tool_name="done", arguments={}))

        return {
            "scenario": scenario,
            "task_idx": task_idx,
            "task": task_description,
            "reward": reward,
            "reward_type": reward_type,
        }


async def main():
    parser = argparse.ArgumentParser(description="Run AWM tasks with expert advisor")
    parser.add_argument("scenario", nargs="?", default="workflow_automation_1")
    parser.add_argument("--baseline-log", required=True, help="Path to baseline log file")
    parser.add_argument("--agent-model", default="gpt-4.1-mini", help="Model for the task agent")
    parser.add_argument("--expert-model", default="gpt-5.1", help="Model for the expert advisor")
    args = parser.parse_args()

    llm = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview"),
    )

    expert = ExpertAdvisor(
        baseline_log_path=args.baseline_log,
        llm=llm,
        model=args.expert_model,
    )

    print(f"Expert model: {args.expert_model}")
    print(f"Agent model:  {args.agent_model}")
    print(f"Baseline log: {args.baseline_log}")
    print(f"Scenario:     {args.scenario}")
    print("=" * 80)

    results = []
    for task_idx in range(10):
        try:
            r = await run_task(
                args.scenario,
                task_idx,
                expert,
                agent_llm=llm,
                agent_model=args.agent_model,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {args.scenario} task {task_idx}: {e}")
            results.append({
                "scenario": args.scenario,
                "task_idx": task_idx,
                "task": "",
                "reward": 0.0,
                "reward_type": "error",
                "error": str(e),
            })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    completed = sum(1 for r in results if r["reward_type"] == "complete")
    print(f"Scenario: {args.scenario}")
    print(f"Expert model: {args.expert_model}")
    print(f"Agent model:  {args.agent_model}")
    print(f"Tasks run: {len(results)}")
    print(f"Completed (reward=1.0): {completed}/{len(results)}")
    print(f"Average reward: {sum(r['reward'] for r in results) / len(results):.2f}")
    print("-" * 80)
    for r in results:
        status = "PASS" if r["reward_type"] == "complete" else "FAIL"
        print(f"  [{status}] Task {r['task_idx']}: {r['reward_type']} (reward={r['reward']}) - {r.get('task','')[:80]}")

    out_file = f"awm_{args.scenario}_expert_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
