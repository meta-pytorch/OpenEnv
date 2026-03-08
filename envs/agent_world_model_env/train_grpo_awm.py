"""
Minimal GRPO training script for Agent World Model using TRL + OpenEnv.

Trains a small LLM to use MCP tools via reinforcement learning, with an
optional GPT expert-in-the-loop for scaffolded learning.

Requirements:
    pip install trl[vllm] openenv-core[core] openai datasets

Usage:
    # 1. Start AWM environment server
    uvicorn agent_world_model_env.server.app:app --host 127.0.0.1 --port 8899

    # 2. Run training (single GPU, colocate vLLM)
    python train_grpo_awm.py

    # Or with server-mode vLLM (2 GPUs):
    # Terminal 1: trl vllm-serve --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
    # Terminal 2: python train_grpo_awm.py --vllm-mode server --vllm-server-url http://localhost:8000
"""

import argparse
import asyncio
import json
import os
import re
from typing import Optional

from datasets import Dataset
from openai import AsyncAzureOpenAI
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


# ---------------------------------------------------------------------------
# System prompt (matches our training setup)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an MCP tool-use agent. You interact with an environment via tool calls.
At each step, call exactly ONE tool using XML tags:

<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "TOOL_NAME", "arguments": "{\\"param\\": \\"value\\"}"}}
</tool_call>

Available meta-tools:
1. list_tools — discover environment tools
2. call_tool — call a specific tool
3. ask_expert — get help from an expert (optional, max 3 calls)

WORKFLOW:
1. Call list_tools to discover available tools
2. TRY to solve the task yourself using call_tool
3. If you get stuck or hit an error, call ask_expert for guidance
4. Follow the expert's plan using call_tool

When done, output your final answer as plain text (no tool_call tags)."""


EXPERT_PROMPT = """\
You are an expert advisor for MCP tool-use agents. Given a task, available tools,
and optional error context, produce a precise step-by-step plan.

Respond with a numbered plan using exact tool names and argument values.
Be specific — the agent will follow your instructions literally."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_tool_call(text: str) -> Optional[dict]:
    """Extract the first <tool_call> block from model output."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
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
    """Format Tool objects into a readable string."""
    lines = [f"Available MCP Tools ({len(tools)} tools):", "=" * 50]
    for i, t in enumerate(tools, 1):
        lines.append(f"{i}. {t.name}: {t.description}")
        props = t.input_schema.get("properties", {})
        if props:
            for pname, pinfo in props.items():
                lines.append(f"   - {pname}: {pinfo.get('type', 'any')} — {pinfo.get('description', '')}")
    return "\n".join(lines)


async def call_expert(task: str, tools_text: str, context: str = "") -> str:
    """Call GPT expert for a step-by-step plan."""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    model = os.environ.get("EXPERT_MODEL", "gpt-4o")

    if endpoint:
        client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview"),
        )
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

    user_msg = f"Task: {task}\n\nAvailable tools:\n{tools_text}"
    if context:
        user_msg += f"\n\nContext/errors so far:\n{context}"

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXPERT_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Environment rollout
# ---------------------------------------------------------------------------
def run_episode(
    env_url: str,
    scenario: str,
    task_idx: int,
    completion_text: str,
    use_expert: bool = False,
    max_steps: int = 15,
) -> dict:
    """
    Run a single agent episode in the AWM environment.

    Takes a model completion (which should contain tool calls), executes them
    in the environment, and returns the reward.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _run_episode_async(env_url, scenario, task_idx, completion_text, use_expert, max_steps)
        )
    finally:
        loop.close()


async def _run_episode_async(
    env_url: str,
    scenario: str,
    task_idx: int,
    completion_text: str,
    use_expert: bool,
    max_steps: int,
) -> dict:
    """Async implementation of episode runner."""
    result_data = {
        "reward": -1.0,
        "classification": "format_error",
        "used_expert": False,
        "steps": 0,
    }

    try:
        async with AWMEnv(base_url=env_url) as env:
            reset_result = await env.reset(scenario=scenario, task_idx=task_idx)
            task_desc = reset_result.observation.task or ""

            tools_result = await env.step(ListToolsAction())
            tools = tools_result.observation.tools
            tools_text = format_tools(tools)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task_desc},
            ]

            expert_calls = 0
            steps_taken = 0

            # Use the model's completion as the first response
            content = completion_text
            messages.append({"role": "assistant", "content": content})

            for step in range(max_steps):
                tc = parse_tool_call(content)
                if not tc:
                    break

                name = tc["name"]
                arguments = tc.get("arguments", {})
                steps_taken += 1

                if name == "list_tools":
                    tool_response = tools_text
                elif name == "ask_expert" and use_expert and expert_calls < 3:
                    expert_calls += 1
                    result_data["used_expert"] = True
                    context = arguments.get("context", "")
                    plan = await call_expert(task_desc, tools_text, context)
                    tool_response = f"Expert plan:\n{plan}"
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

                    step_result = await env.step(
                        CallToolAction(tool_name=tool_name, arguments=inner_args)
                    )
                    obs = step_result.observation
                    if hasattr(obs, "tool_result") and obs.tool_result is not None:
                        tool_response = (
                            json.dumps(obs.tool_result, ensure_ascii=False)
                            if not isinstance(obs.tool_result, str)
                            else obs.tool_result
                        )
                    elif hasattr(obs, "error") and obs.error:
                        tool_response = f"Error: {obs.error}"
                    else:
                        tool_response = str(obs.model_dump())
                else:
                    tool_response = f"Error: Unknown tool '{name}'."

                messages.append({"role": "user", "content": f"Tool response:\n{tool_response}"})
                # For multi-turn, we'd generate another completion here.
                # In this minimal version, we only use the first completion.
                break

            # Verify
            verify_result = await env.step(
                CallToolAction(
                    tool_name="verify",
                    arguments={"verifier_mode": "code", "final_answer": content},
                )
            )
            reward = verify_result.reward or 0.0
            reward_type = getattr(verify_result.observation, "reward_type", "unknown")

            result_data["reward"] = float(reward)
            result_data["classification"] = reward_type
            result_data["steps"] = steps_taken

            await env.step(CallToolAction(tool_name="done", arguments={}))

    except Exception as e:
        result_data["reward"] = -1.0
        result_data["classification"] = "error"
        result_data["error"] = str(e)

    return result_data


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------
def reward_from_env(completions, **kwargs):
    """Extract environment reward passed from rollout."""
    rewards = kwargs.get("env_reward", [])
    if rewards:
        return [float(r) for r in rewards]
    return [0.0] * len(completions)


def reward_solo_bonus(completions, **kwargs):
    """Bonus for completing tasks without expert."""
    used_expert = kwargs.get("used_expert", [])
    classifications = kwargs.get("classification", [])
    rewards = []
    for i in range(len(completions)):
        is_complete = classifications[i] == "complete" if i < len(classifications) else False
        did_use_expert = used_expert[i] if i < len(used_expert) else False
        rewards.append(1.0 if is_complete and not did_use_expert else 0.0)
    return rewards


def reward_format_penalty(completions, **kwargs):
    """Penalize malformed tool calls."""
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else str(c)
        has_valid_call = parse_tool_call(text) is not None
        has_no_call = "<tool_call>" not in text
        # Penalize if the model tried but failed to format a tool call
        if not has_valid_call and not has_no_call:
            rewards.append(-0.5)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------
def make_rollout_func(env_url: str, scenarios: list, use_expert: bool = True):
    """Create a rollout function that interacts with the AWM environment."""

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict:
        outputs = generate_rollout_completions(trainer, prompts)
        tokenizer = trainer.processing_class

        prompt_ids = [out["prompt_ids"] for out in outputs]
        completion_ids = [out["completion_ids"] for out in outputs]
        logprobs = [out["logprobs"] for out in outputs]

        completions_text = [
            tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
            for out in outputs
        ]

        env_rewards = []
        used_expert_list = []
        classifications = []

        for i, completion in enumerate(completions_text):
            scenario_info = scenarios[i % len(scenarios)]
            scenario_name = scenario_info["scenario"]
            task_idx = scenario_info["task_idx"]

            # Alternate: half with expert, half without
            episode_use_expert = use_expert and (i % 2 == 0)

            result = run_episode(
                env_url=env_url,
                scenario=scenario_name,
                task_idx=task_idx,
                completion_text=completion,
                use_expert=episode_use_expert,
                max_steps=10,
            )

            env_rewards.append(result["reward"])
            used_expert_list.append(result["used_expert"])
            classifications.append(result["classification"])

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "env_reward": env_rewards,
            "used_expert": used_expert_list,
            "classification": classifications,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------
def create_training_dataset(scenarios: list) -> Dataset:
    """Create a dataset of prompts from AWM scenarios."""
    prompts = []
    for s in scenarios:
        task = s.get("task", s.get("question", "Complete the task using the available tools."))
        prompt = f"{SYSTEM_PROMPT}\n\nTask: {task}\n\nStart by calling list_tools to discover available tools."
        prompts.append(prompt)
    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GRPO training on Agent World Model")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model to train")
    parser.add_argument("--env-url", default="http://localhost:8899", help="AWM server URL")
    parser.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"])
    parser.add_argument("--vllm-server-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="./output_awm_grpo")
    parser.add_argument("--num-generations", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-steps", type=int, default=50, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--use-expert", action="store_true", help="Enable GPT expert tool")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Scenario names (default: sample workflow automation)")
    args = parser.parse_args()

    # Build scenario list
    if args.scenarios:
        scenarios = [{"scenario": s, "task_idx": 0} for s in args.scenarios]
    else:
        scenarios = [
            {"scenario": "flowlatch_automation_studio", "task_idx": i, "task":
             f"Complete workflow automation task {i} using the available MCP tools."}
            for i in range(5)
        ]

    # Load actual tasks from the environment if possible
    try:
        from agent_world_model_env.server.data_loader import AWMDataLoader
        loader = AWMDataLoader()
        enriched = []
        for s in scenarios:
            tasks = loader.get_tasks(s["scenario"])
            if tasks and s["task_idx"] < len(tasks):
                s["task"] = tasks[s["task_idx"]]
            enriched.append(s)
        scenarios = enriched
        print(f"Loaded {len(scenarios)} tasks from AWM dataset")
    except Exception as e:
        print(f"Could not load tasks from dataset: {e}, using defaults")

    dataset = create_training_dataset(scenarios)
    print(f"Training dataset: {len(dataset)} prompts")
    print(f"Model: {args.model}")
    print(f"Expert: {'enabled' if args.use_expert else 'disabled'}")

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        num_generations=args.num_generations,
        max_completion_length=2048,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=5e-6,
        temperature=1.0,
        beta=0.001,
        logging_steps=1,
        save_steps=10,
        report_to="none",
    )

    rollout_fn = make_rollout_func(
        env_url=args.env_url,
        scenarios=scenarios,
        use_expert=args.use_expert,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_from_env, reward_solo_bonus, reward_format_penalty],
        train_dataset=dataset,
        rollout_func=rollout_fn,
        args=grpo_config,
    )

    print("Starting GRPO training...")
    trainer.train()
    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
