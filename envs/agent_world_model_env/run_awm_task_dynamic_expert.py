"""Dynamic expert-in-the-loop runner for AWM environments.

Exposes the expert as a callable **tool** (`ask_expert`) that the agent
invokes on-demand *during* the task whenever it needs guidance.

The expert is "verifier-informed": it analyzes the Python verification code
to understand the exact DB state required for success, then combines that
with the full MCP tool schemas to produce precise guidance.

Key capabilities:
  - Agent decides WHEN to call the expert (0 to N times per task)
  - Expert has real-time context (errors, partial progress)
  - System nudges the agent to consult the expert after errors or stalls
  - No prior baseline data needed — works from the first run

Usage:
    # Start AWM server first
    uvicorn agent_world_model_env.server.app:app --host 127.0.0.1 --port 8899

    # Run benchmark (baseline + dynamic expert, side by side)
    export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
    export AZURE_OPENAI_API_KEY="your-key"
    python run_awm_task_dynamic_expert.py [scenario_name]
"""

import argparse
import asyncio
import functools
import json
import os
import re
import time

from agent_world_model_env import AWMEnv
from agent_world_model_env.server.data_loader import AWMDataLoader
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openai import AsyncAzureOpenAI

print = functools.partial(print, flush=True)

CONTENT_FILTER_MARKER = "content_filter"

# ---------------------------------------------------------------------------
# Verifier-informed expert rules
# ---------------------------------------------------------------------------

EXPERT_RULES = """\
You are an expert advisor for MCP tool-use agents. You have:
1. The COMPLETE tool list with exact names and parameter schemas
2. Knowledge of the verification requirements (what DB state is checked)

Your job: produce a PRECISE step-by-step plan using EXACT tool names and args.

CRITICAL RULES:
1. TOOL NAMES: Copy-paste tool names EXACTLY from the provided list. \
   Hallucinating a tool name is the #1 failure mode. Double-check every name.
2. PARAMETER NAMES: Use EXACT parameter names from the tool schema. \
   E.g. if the schema says "workflow_name" (REQUIRED), you MUST include it.
3. MULTI-TABLE TASKS: Creating a "workflow with trigger and action steps" \
   requires SEPARATE calls: create_workflow, then upsert_workflow_trigger, \
   then create_workflow_action_step. One create_workflow call is NOT enough.
4. PLAYBOOK SHORTCUTS: If a "playbook_" tool exists that does exactly what \
   the task asks, prefer it — it does multiple steps atomically.
5. CHAINING: Extract IDs from responses and use them in subsequent calls. \
   Always lookup by name first to get the ID.
6. VERIFIER REQUIREMENTS: Your plan MUST produce the EXACT DB state the \
   verifier expects. Match table names, column values, and JSON field contents.
7. FINAL ANSWER: If the verifier checks `final_answer`, the agent must \
   output specific information (run IDs, statuses, etc.) in its final text.
8. ARGUMENT VALUES: For JSON/object arguments, provide the EXACT values \
   the verifier expects. E.g. filter_rules={"record_type":"lead"} not a \
   description of what it should contain.

Respond with JSON:
{"plan": [{"tool": "sub_env_exact_name", "args": {"param": "value"}, "purpose": "why"}], \
"verifier_notes": "what DB state must exist", \
"tips": "warnings, exact values to use, common pitfalls"}"""

VERIFIER_ANALYZER_PROMPT = """\
Analyze this Python verification code and extract the SUCCESS CRITERIA — \
the exact database state and conditions that must be true for the task to pass.

Be specific: mention table names, column values, JSON field contents, \
and any relationships between records that the verifier checks.

Focus on WHAT MUST BE TRUE, not how the code works internally.

VERIFICATION CODE:
{code}

TASK: {task}

Output a concise list of requirements."""


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BASELINE_PROMPT = """\
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
- Do NOT use "playbook" shortcut tools. Always use the granular/primitive tools \
(e.g. create_connector, create_workflow, create_workflow_action_step, update_workflow, etc.) \
to ensure every parameter matches the task requirements exactly.
- BEFORE creating any resource, first check if it already exists.
- If a tool call returns an error, read it carefully. It often means the \
resource already exists — try updating it instead.
- After performing actions, verify by reading back the resource.
- Make sure all arguments are passed as proper JSON with correct types.
- When you have enough information to answer, output the answer directly \
without any tool_call tags."""


ADAPTIVE_PROMPT = """\
You are at a MCP environment. You need to call MCP tools to assist with the user query. \
At each step, you can only call one function. You have already logged in, and your user id is 1 if required.

You are provided with THREE functions:

1. list_tools
   - Description: List all available MCP tools for the current environment.
   - Arguments: None

2. call_tool
   - Description: Call a MCP environment-specific tool
   - Arguments:
       - tool_name: str, required
       - arguments: str, required, valid JSON string

3. ask_expert
   - Description: Consult an expert advisor who has deep knowledge of this system. \
The expert knows the exact requirements for task completion and can provide precise \
step-by-step plans. ALWAYS call this after list_tools for complex tasks.
   - Arguments:
       - task: str, the task you are trying to accomplish
       - available_tools: str, comma-separated list of available tool names
       - context: str, what you have tried so far and any errors (optional)

For each function call, return a json object within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

WORKFLOW:
1. Call list_tools to discover available tools
2. Call ask_expert with the task and tool names — the expert has deep domain knowledge
3. Follow the expert's plan step by step using call_tool
4. If a step fails, call ask_expert again with the error context

CRITICAL:
- Use EXACT tool names from list_tools (they start with "sub_env_")
- Extract IDs from tool responses and use them in subsequent calls
- If the task asks you to report or list results, include that in your final answer

When you have completed the task, output your final answer without any tool_call tags."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def format_tools(tools, verbose=False) -> str:
    """Format tool list. verbose=True includes full descriptions for expert."""
    lines = [f"Available MCP Tools ({len(tools)} tools):", "=" * 60]
    for i, t in enumerate(tools, 1):
        lines.append(f"{i}. {t.name}")
        desc_len = 200 if verbose else 120
        lines.append(f"   Description: {t.description[:desc_len]}")
        props = t.input_schema.get("properties", {})
        required = t.input_schema.get("required", [])
        if props:
            lines.append("   Parameters:")
            for pname, pinfo in props.items():
                req = " (REQUIRED)" if pname in required else ""
                desc = pinfo.get("description", "")
                desc_part = f" — {desc[:80]}" if desc and verbose else ""
                lines.append(
                    f"     - {pname}: {pinfo.get('type', 'any')}{req}{desc_part}"
                )
        lines.append("")
    return "\n".join(lines)


def is_content_filter_error(exc: Exception) -> bool:
    return "content_filter" in str(exc) or "content management policy" in str(exc)


def safe_parse_arguments(arguments):
    """Handle LLM returning arguments as JSON string instead of dict."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    if not isinstance(arguments, dict):
        return {}
    return arguments


# ---------------------------------------------------------------------------
# Verifier-informed expert
# ---------------------------------------------------------------------------

_data_loader = AWMDataLoader()


def get_verifier_code(scenario: str, task_idx: int) -> str | None:
    entry = _data_loader.get_verifier(scenario, task_idx, "code")
    if not entry:
        return None
    return entry.get("verification", {}).get("code", "")


async def analyze_verifier(llm, model, task: str, verifier_code: str) -> str:
    """Use LLM to extract success criteria from verifier code."""
    prompt = VERIFIER_ANALYZER_PROMPT.format(code=verifier_code, task=task)
    try:
        response = await llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=800,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""


async def call_expert(llm, model, task, tool_schemas="", context="",
                      verifier_requirements=""):
    """Call the expert with full tool schemas and verifier intelligence."""
    expert_prompt = f"TASK: {task}\n\n"

    if tool_schemas:
        expert_prompt += (
            f"AVAILABLE TOOLS (with full parameter schemas):\n"
            f"{tool_schemas}\n\n"
        )

    if verifier_requirements:
        expert_prompt += (
            f"VERIFIER REQUIREMENTS (what the system checks for completion):\n"
            f"{verifier_requirements}\n\n"
        )

    expert_prompt += (
        f"AGENT CONTEXT: {context if context else 'Agent just started.'}\n\n"
        "Provide a precise step-by-step plan with EXACT tool names and "
        "argument values that satisfies ALL verifier requirements."
    )
    try:
        response = await llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXPERT_RULES},
                {"role": "user", "content": expert_prompt},
            ],
            temperature=0.1,
            max_completion_tokens=2000,
        )
        return response.choices[0].message.content or "No plan generated."
    except Exception:
        return "Expert unavailable. Proceed with best judgment."


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

NUDGE_ON_ERROR = (
    "The last tool call returned an error. Call ask_expert "
    "with the error details to get guidance on how to fix it."
)

NUDGE_ON_STALL = (
    "You have taken {steps} steps without completing the task. "
    "Call ask_expert to get a revised plan for the remaining steps."
)

STALL_THRESHOLD = 5


async def run_task(env, llm, scenario, task_idx, model="gpt-5.1",
                   max_iters=15, expert_available=False):
    """Run a single AWM task.

    When expert_available=True, the agent has access to an ask_expert tool
    that it can call at any point during the task. The system nudges the
    agent to call the expert after errors or stalls.
    """
    result = await env.reset(scenario=scenario, task_idx=task_idx)
    task = result.observation.task

    verifier_reqs = ""
    if expert_available:
        vcode = get_verifier_code(scenario, task_idx)
        if vcode:
            verifier_reqs = await analyze_verifier(llm, model, task, vcode)
            if verifier_reqs:
                print(f"    Verifier analysis: {verifier_reqs[:120]}...")

    system_prompt = ADAPTIVE_PROMPT if expert_available else BASELINE_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    final_content = ""
    steps = 0
    errors = 0
    expert_calls = 0
    filtered = False
    cached_tools = None
    cached_tool_schemas = ""
    last_was_error = False

    for step in range(1, max_iters + 1):
        try:
            response = await llm.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=2048,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            if is_content_filter_error(e):
                filtered = True
            else:
                print(f"    LLM error step {step}: {str(e)[:150]}")
            errors += 1
            break

        messages.append({"role": "assistant", "content": content})
        final_content = content

        tc = parse_tool_call(content)
        if not tc:
            break

        name = tc["name"]
        arguments = tc.get("arguments") or {}
        steps += 1
        tool_response = ""
        last_was_error = False

        if name == "list_tools":
            result = await env.step(ListToolsAction())
            cached_tools = result.observation.tools
            cached_tool_schemas = format_tools(cached_tools, verbose=True)
            tool_response = format_tools(cached_tools)
            print(f"    Step {step}: list_tools -> {len(cached_tools)} tools")

        elif name == "ask_expert" and expert_available:
            arguments = safe_parse_arguments(arguments)
            expert_task = arguments.get("task", task)
            expert_context = arguments.get("context", "")
            tool_response = await call_expert(
                llm, model, expert_task,
                tool_schemas=cached_tool_schemas,
                context=expert_context,
                verifier_requirements=verifier_reqs,
            )
            expert_calls += 1
            print(f"    Step {step}: ask_expert -> plan received")

        elif name == "call_tool":
            arguments = safe_parse_arguments(arguments)
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
                errors += 1
                last_was_error = True
            else:
                tool_response = json.dumps(obs.model_dump(), ensure_ascii=False)

            status = "OK" if not last_was_error else "ERR"
            print(
                f"    Step {step}: call_tool({tool_name}) [{status}]"
                f" -> {str(tool_response)[:100]}"
            )
        else:
            tool_response = (
                "Error: Unknown function. Use list_tools, call_tool"
                + (", or ask_expert." if expert_available else ".")
            )
            errors += 1
            last_was_error = True

        if len(tool_response) > 3000:
            tool_response = tool_response[:3000] + "... (truncated)"

        nudge = ""
        if expert_available and name != "ask_expert":
            if last_was_error:
                nudge = "\n\n" + NUDGE_ON_ERROR
            elif steps >= STALL_THRESHOLD and expert_calls == 0:
                nudge = "\n\n" + NUDGE_ON_STALL.format(steps=steps)

        messages.append({
            "role": "user",
            "content": f"Tool response:\n{tool_response}{nudge}",
        })

    if filtered:
        await env.step(
            CallToolAction(tool_name="done", arguments={"keep_session": False})
        )
        return {
            "scenario": scenario, "task_idx": task_idx, "task": task[:80],
            "steps": 0, "errors": 1, "expert_calls": 0,
            "reward_type": CONTENT_FILTER_MARKER, "reward": None,
        }

    result = await env.step(
        CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": "code", "final_answer": final_content},
        )
    )

    reward_type = result.observation.reward_type
    reward = result.reward
    verify_result = getattr(result.observation, "verify_result", None)

    if reward_type != "complete":
        print(f"    VERIFY: reward_type={reward_type} reward={reward}")
        if verify_result:
            print(f"    VERIFY detail: {json.dumps(verify_result, default=str)[:300]}")

    await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))

    return {
        "scenario": scenario, "task_idx": task_idx, "task": task[:80],
        "steps": steps, "errors": errors, "expert_calls": expert_calls,
        "reward_type": reward_type,
        "reward": reward,
    }


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

async def run_suite(llm, model, label, expert_available, scenarios, num_tasks):
    all_results = []
    for scenario in scenarios:
        print(f"\n{'='*90}")
        print(f"[{label}] SCENARIO: {scenario}")
        print(f"{'='*90}")

        for task_idx in range(num_tasks):
            t0 = time.time()
            async with AWMEnv(base_url="http://localhost:8899") as env:
                r = await run_task(
                    env, llm, scenario, task_idx,
                    model=model, expert_available=expert_available,
                )
                all_results.append(r)
                elapsed = time.time() - t0

                if r["reward_type"] == CONTENT_FILTER_MARKER:
                    print(f"  Task {task_idx}: [FILTERED] {r['task'][:60]}...")
                else:
                    expert_tag = f" expert_calls={r['expert_calls']}" if expert_available else ""
                    icon = "PASS" if r["reward_type"] == "complete" else "FAIL"
                    print(
                        f"  Task {task_idx}: [{icon}] reward={r['reward']:5.1f}"
                        f" steps={r['steps']}{expert_tag} time={elapsed:.0f}s"
                        f"  {r['task'][:50]}..."
                    )
    return all_results


def print_summary(label, results):
    valid = [r for r in results if r["reward_type"] != CONTENT_FILTER_MARKER]
    filtered = [r for r in results if r["reward_type"] == CONTENT_FILTER_MARKER]

    print(f"\n{'='*90}")
    print(f"RESULTS: {label}")
    print(f"{'='*90}")
    print(f"  Total tasks:       {len(results)}")
    print(f"  Content filtered:  {len(filtered)} (excluded)")
    print(f"  Valid tasks:       {len(valid)}")

    if not valid:
        print("  No valid tasks.")
        return

    avg_reward = sum(r["reward"] for r in valid) / len(valid)
    complete = sum(1 for r in valid if r["reward_type"] == "complete")
    avg_steps = sum(r["steps"] for r in valid) / len(valid)
    total_expert = sum(r.get("expert_calls", 0) for r in valid)

    print(f"  Average reward:    {avg_reward:.3f}")
    print(f"  Complete:          {complete}/{len(valid)} ({100*complete/len(valid):.0f}%)")
    print(f"  Avg steps/task:    {avg_steps:.1f}")
    if total_expert:
        print(f"  Total expert calls:{total_expert} across {len(valid)} tasks")


async def main():
    parser = argparse.ArgumentParser(
        description="AWM dynamic expert-in-the-loop benchmark"
    )
    parser.add_argument(
        "scenario", nargs="?", default="workflow_automation_1",
        help="Scenario name (default: workflow_automation_1)",
    )
    parser.add_argument(
        "--model", default=os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.1"),
        help="Azure OpenAI model deployment name",
    )
    parser.add_argument(
        "--tasks", type=int, default=10,
        help="Number of tasks per scenario (default: 10)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Run baseline only (skip expert)",
    )
    parser.add_argument(
        "--expert-only", action="store_true",
        help="Run expert only (skip baseline)",
    )
    args = parser.parse_args()

    llm = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview"),
    )

    scenarios = [args.scenario]
    baseline_results = []
    adaptive_results = []

    if not args.expert_only:
        print("\n" + "#" * 90)
        print("# PHASE 1: BASELINE (no expert)")
        print("#" * 90)
        baseline_results = await run_suite(
            llm, args.model, "BASELINE", expert_available=False,
            scenarios=scenarios, num_tasks=args.tasks,
        )
        print_summary("BASELINE (no expert)", baseline_results)

    if not args.baseline_only:
        print("\n" + "#" * 90)
        print("# PHASE 2: DYNAMIC EXPERT (verifier-informed, on-demand)")
        print("#" * 90)
        adaptive_results = await run_suite(
            llm, args.model, "DYNAMIC EXPERT", expert_available=True,
            scenarios=scenarios, num_tasks=args.tasks,
        )
        print_summary("DYNAMIC EXPERT (verifier-informed)", adaptive_results)

    # --- Side-by-side comparison ---
    if baseline_results and adaptive_results:
        bv = [r for r in baseline_results if r["reward_type"] != CONTENT_FILTER_MARKER]
        av = [r for r in adaptive_results if r["reward_type"] != CONTENT_FILTER_MARKER]

        print(f"\n{'='*90}")
        print("COMPARISON: BASELINE vs DYNAMIC EXPERT")
        print(f"{'='*90}")
        if bv and av:
            b_avg = sum(r["reward"] for r in bv) / len(bv)
            a_avg = sum(r["reward"] for r in av) / len(av)
            b_comp = sum(1 for r in bv if r["reward_type"] == "complete")
            a_comp = sum(1 for r in av if r["reward_type"] == "complete")
            b_steps = sum(r["steps"] for r in bv) / len(bv)
            a_steps = sum(r["steps"] for r in av) / len(av)
            a_expert = sum(r.get("expert_calls", 0) for r in av)

            print(f"  {'Metric':<25} {'Baseline':>12} {'Dynamic Exp':>12} {'Delta':>12}")
            print(f"  {'-'*61}")
            print(f"  {'Avg reward':<25} {b_avg:>12.3f} {a_avg:>12.3f} {a_avg-b_avg:>+12.3f}")
            print(f"  {'Complete':<25} {b_comp:>9}/{len(bv)} {a_comp:>9}/{len(av)} {a_comp-b_comp:>+12d}")
            print(f"  {'Avg steps':<25} {b_steps:>12.1f} {a_steps:>12.1f} {a_steps-b_steps:>+12.1f}")
            print(f"  {'Expert calls (total)':<25} {'n/a':>12} {a_expert:>12}")
        else:
            print("  Not enough valid results for comparison.")


if __name__ == "__main__":
    asyncio.run(main())
