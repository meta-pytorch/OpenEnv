#!/usr/bin/env python3
"""Play FinQA with an LLM via any OpenAI-compatible API.

FinQA is a financial question-answering benchmark that evaluates LLMs on their
ability to answer complex financial questions using tool calls on SEC 10-K filing data.

The agent must:
1. Explore available tables for a company
2. Query table metadata and execute SQL queries
3. Perform calculations on extracted data
4. Submit final answers to financial questions

Prerequisites
-------------
1. Build the FinQA Docker image::

       docker build -f src/envs/finqa_env/server/Dockerfile -t finqa-env:latest .

2. Set your API key::

       export HF_TOKEN=your_token_here

3. Run this script::

       python examples/finqa_inference.py

To use a different provider, set API_BASE_URL and MODEL::

       export API_BASE_URL=https://api.openai.com/v1
       export MODEL=gpt-4o
       export API_KEY=your_openai_key
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.finqa_env import FinQAAction, FinQAEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b:novita")

MAX_EPISODES = 3
MAX_TOKENS = 2048
VERBOSE = True

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about SEC 10-K filings.

Think and reason step by step. Iteratively gather data using the available tools until you have enough information to answer the question.

When submitting your final answer, provide ONLY the numerical value (e.g., '6.118', '92.61%', '-77'). Do not include explanations or units in the answer.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_tools_from_server(base_url: str) -> List[dict]:
    """Fetch tool schemas from the FinQA server."""
    import requests

    response = requests.get(f"{base_url}/tools")
    response.raise_for_status()
    return response.json()


def make_initial_messages(company: str, question: str) -> List[Dict[str, Any]]:
    """Create initial chat messages for a FinQA episode."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Company: {company}\nQuestion: {question}"},
    ]


# ---------------------------------------------------------------------------
# Gameplay
# ---------------------------------------------------------------------------


def play_finqa_episode(
    env: FinQAEnv,
    client: OpenAI,
    tools: List[dict],
    episode_num: int,
) -> Dict[str, Any]:
    """Play a single FinQA episode."""
    tool_names = [t["function"]["name"] for t in tools]

    result = env.reset()
    obs = result.observation

    if VERBOSE:
        print(f"\n{'=' * 60}")
        print(f"Episode {episode_num}")
        print(f"{'=' * 60}")
        print(f"Company: {obs.company}")
        print(f"Question: {obs.question}")

    # Maintain chat history for multi-turn conversation
    chat_history = make_initial_messages(obs.company, obs.question)
    tool_calls = []

    while not result.done:
        if VERBOSE:
            print(f"\n--- Step {obs.step_count + 1} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=chat_history,
            tools=tools,
            tool_choice="required",
            max_completion_tokens=MAX_TOKENS,
        )

        message = response.choices[0].message

        # Handle case where model doesn't return a tool call
        if not message.tool_calls:
            if VERBOSE:
                print(f"No tool call returned, submitting text response")
            tool_name = "submit_answer"
            tool_args = {"answer": message.content or "unknown"}
            tool_call_id = "none"
            tool_args_raw = json.dumps(tool_args)
        else:
            tool_call_obj = message.tool_calls[0]
            tool_name = tool_call_obj.function.name
            tool_args = json.loads(tool_call_obj.function.arguments)
            tool_call_id = tool_call_obj.id
            tool_args_raw = tool_call_obj.function.arguments

        chat_history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_args_raw},
                    }
                ],
            }
        )

        if VERBOSE:
            print(f"Tool: {tool_name}({json.dumps(tool_args)})"[:100])

        # End conversation if model uses invalid tool
        if tool_name not in tool_names:
            if VERBOSE:
                print(f"  Invalid tool '{tool_name}', submitting unknown")
            tool_name = "submit_answer"
            tool_args = {"answer": "unknown"}

        action = FinQAAction(tool_name=tool_name, tool_args=tool_args)
        result = env.step(action)
        obs = result.observation

        tool_calls.append(
            {
                "tool": tool_name,
                "args": tool_args,
                "result": obs.tool_result or "",
            }
        )

        if not result.done:
            chat_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": obs.tool_result or "No result",
                }
            )

        if VERBOSE:
            result_preview = (obs.tool_result or "")[:200]
            print(f"Result: {result_preview}...")

    # Get ground truth from environment state
    state = env.state()
    reward = result.reward or 0.0

    submitted_answer = None
    for tc in reversed(tool_calls):
        if tc.get("tool") == "submit_answer":
            submitted_answer = tc.get("args", {}).get("answer")
            break

    if VERBOSE:
        outcome = "CORRECT" if reward > 0 else "INCORRECT"
        print(f"\nResult: {outcome}")
        print(f"  Submitted: {submitted_answer}")
        print(f"  Ground truth: {state.ground_truth}")
        print(f"  Reward: {reward}")

    return {
        "episode": episode_num,
        "company": obs.company,
        "question": obs.question,
        "submitted_answer": submitted_answer,
        "ground_truth": state.ground_truth,
        "reward": reward,
        "steps": obs.step_count,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        raise SystemExit("API_KEY (or HF_TOKEN) must be set to query the model.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = FinQAEnv.from_docker_image("finqa-env:latest")
    tools = fetch_tools_from_server(env.base_url)

    if VERBOSE:
        tool_names = [t["function"]["name"] for t in tools]
        print(f"API: {API_BASE_URL}")
        print(f"Model: {MODEL}")
        print(f"Tools: {tool_names}")

    results = []

    try:
        for episode_num in range(1, MAX_EPISODES + 1):
            episode_result = play_finqa_episode(env, client, tools, episode_num)
            results.append(episode_result)

        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        correct = sum(1 for r in results if r["reward"] > 0)
        avg_steps = sum(r["steps"] for r in results) / len(results)

        print(f"Episodes: {len(results)}")
        print(
            f"Correct: {correct}/{len(results)} ({100 * correct / len(results):.1f}%)"
        )
        print(f"Average steps: {avg_steps:.1f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
