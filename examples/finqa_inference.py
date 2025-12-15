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

2. Set your API key (depends on provider)::

       export HF_TOKEN=your_token_here      # HuggingFace (default)
       export OPENAI_API_KEY=your_key       # OpenAI
       export TOGETHER_API_KEY=your_key     # Together AI

3. Run this script::

       python examples/finqa_inference.py

Usage:
    # Default: HuggingFace Inference Providers
    python examples/finqa_inference.py

    # OpenAI
    python examples/finqa_inference.py --api-base https://api.openai.com/v1 --model gpt-5-mini

    # vLLM local server
    python examples/finqa_inference.py --api-base http://localhost:8080/v1 --model my-model

    # Together AI
    python examples/finqa_inference.py --api-base https://api.together.xyz/v1 --model meta-llama/Llama-3-70b-chat-hf

    # Ollama
    python examples/finqa_inference.py --api-base http://localhost:11434/v1 --model llama3

    # More options
    python examples/finqa_inference.py --games 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.finqa_env import FinQAAction, FinQAEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5-mini"

MAX_STEPS = 20
VERBOSE = True
TRACE = False  # Show full untruncated output
MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Tool Definitions 
# ---------------------------------------------------------------------------

def make_tool(name: str, description: str, params: Dict[str, tuple]) -> dict:
    """Create OpenAI tool schema from compact definition.

    Args:
        name: Tool name
        description: Tool description
        params: Dict of {param_name: (type, description)} - all params are required
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    param: {"type": ptype, "description": pdesc}
                    for param, (ptype, pdesc) in params.items()
                },
                "required": list(params.keys())
            }
        }
    }

OPENAI_TOOLS = [
    make_tool(
        "get_descriptions",
        "Get a list of available table names for a company.",
        {"company_name": ("string", "The company name")}
    ),
    make_tool(
        "get_table_info",
        "Get table metadata: description, columns, types, unique values.",
        {
            "company_name": ("string", "The company name"),
            "table_name": ("string", "The table name")
        }
    ),
    make_tool(
        "sql_query",
        "Execute SQL query on a table. MUST include WHERE/HAVING filters.",
        {
            "company_name": ("string", "The company name"),
            "table_name": ("string", "The table name"),
            "query": ("string", "SQL query with WHERE clause")
        }
    ),
    make_tool(
        "submit_answer",
        "Submit final answer. Provide ONLY the numerical value.",
        {"answer": ("string", "Numerical answer only (e.g., '6.118', '-77')")}
    ),
]

AVAILABLE_TOOLS = [t["function"]["name"] for t in OPENAI_TOOLS]

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about SEC 10-K filings.

Think and reason step by step. Iteratively gather data using the available tools until you have enough information to answer the question.

When submitting your final answer, provide ONLY the numerical value (e.g., '6.118', '92.61%', '-77'). Do not include explanations or units in the answer.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_history(history: List[Dict[str, Any]]) -> str:
    """Format tool call history for the prompt."""
    if not history:
        return "None yet."

    lines = []
    for entry in history[-5:]:  # Show last 5 tool calls
        tool = entry.get("tool", "unknown")
        args = entry.get("args", {})
        result = entry.get("result", "")

        # Truncate long results
        if len(result) > 500:
            result = result[:500] + "..."

        lines.append(f"  Step {entry.get('step', '?')}: {tool}({json.dumps(args)})")
        lines.append(f"    Result: {result}")

    return "\n".join(lines)


def build_user_prompt(observation) -> str:
    """Build the user prompt from the observation."""
    history_str = format_history(observation.history)

    prompt = f"""Company: {observation.company}
Question: {observation.question}

Step: {observation.step_count + 1}/{MAX_STEPS}
Available tools: {', '.join(observation.available_tools)}

Previous tool calls:
{history_str}
"""

    if observation.tool_result:
        # Truncate very long results
        result = observation.tool_result
        if len(result) > 1000:
            result = result[:1000] + "...[truncated]"
        prompt += f"\nLast tool result:\n{result}\n"

    prompt += "\nWhat tool would you like to call next? Respond with a JSON object."

    return prompt


def parse_tool_call(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse a tool call from the model's response."""
    if not response_text:
        return None

    # Try to parse the entire response as JSON first
    try:
        parsed = json.loads(response_text.strip())
        if isinstance(parsed, dict) and "tool_name" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Find all JSON objects with balanced braces (handles nested objects)
    def find_json_objects(text: str) -> List[str]:
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 1
                start = i
                i += 1
                while i < len(text) and depth > 0:
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(text[start:i])
            else:
                i += 1
        return objects

    # Try each JSON object found
    for obj_str in find_json_objects(response_text):
        try:
            parsed = json.loads(obj_str)
            if isinstance(parsed, dict) and "tool_name" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# Game Play
# ---------------------------------------------------------------------------

def play_finqa_episode(env: FinQAEnv, client: OpenAI, episode_num: int) -> Dict[str, Any]:
    """Play a single FinQA episode."""
    result = env.reset()
    observation = result.observation

    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"Episode {episode_num}")
        print(f"{'='*60}")
        print(f"Company: {observation.company}")
        print(f"Question: {observation.question}")

    tool_calls = []

    # Maintain chat history for multi-turn conversation
    chat_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Company: {observation.company}\nQuestion: {observation.question}\n\nUse the available tools to find the answer."},
    ]

    while not result.done:
        if VERBOSE:
            print(f"--- Step {observation.step_count + 1}/{MAX_STEPS} ---")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=chat_history,
                tools=OPENAI_TOOLS,
                tool_choice="required",  # Force tool use
                max_completion_tokens=MAX_TOKENS,
            )

            message = response.choices[0].message
            if message.tool_calls:
                tool_call_obj = message.tool_calls[0]
                tool_name = tool_call_obj.function.name
                tool_args = json.loads(tool_call_obj.function.arguments)
                tool_call_id = tool_call_obj.id
                chat_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call_obj.function.arguments
                        }
                    }]
                })

                if VERBOSE:
                    if TRACE:
                        print(f"Tool call: {tool_name}({json.dumps(tool_args, indent=2)})")
                    else:
                        print(f"Tool call: {tool_name}({json.dumps(tool_args)})"[:200])

        except Exception as e:
            print(f"Error calling model: {e}")
            # Default to submitting "unknown" if model fails
            tool_name = "submit_answer"
            tool_args = {"answer": "unknown"}
            tool_call_id = "error"

        if tool_name not in AVAILABLE_TOOLS:
            if VERBOSE:
                print(f"  Invalid tool '{tool_name}', submitting unknown")
            tool_name = "submit_answer"
            tool_args = {"answer": "unknown"}

        if VERBOSE:
            print(f"  Tool: {tool_name}")
            print(f"  Args: {tool_args}")

        action = FinQAAction(tool_name=tool_name, tool_args=tool_args)
        result = env.step(action)
        observation = result.observation

        tool_calls.append({
            "tool": tool_name,
            "args": tool_args,
            "result": observation.tool_result[:200] if observation.tool_result else "",
        })

        if not result.done:
            chat_history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": observation.tool_result or "No result"
            })

        if VERBOSE:
            if TRACE:
                print(f"  Result:\n{observation.tool_result}\n")
            else:
                result_preview = observation.tool_result[:200] if observation.tool_result else ""
                print(f"  Result: {result_preview}...")
            print()

    reward = result.reward or 0.0
    ground_truth = observation.metadata.get("ground_truth") if observation.metadata else None
    submitted_answer = None
    for tc in reversed(tool_calls):
        if tc.get("tool") == "submit_answer":
            submitted_answer = tc.get("args", {}).get("answer")
            break

    if VERBOSE:
        outcome = "CORRECT" if reward > 0 else "INCORRECT"
        print(f"Episode {episode_num} Result: {outcome}")
        print(f"  Submitted: {submitted_answer}")
        print(f"  Ground truth: {ground_truth}")
        print(f"  Reward: {reward}")
        print(f"  Steps: {observation.step_count}")

    if TRACE:
        print(f"\n{'='*60}")
        print(f"FULL TRACE - Episode {episode_num}")
        print(f"{'='*60}")
        print(f"Question: {observation.question}")
        print(f"Company: {observation.company}")
        print(f"\nTool calls:")
        for i, entry in enumerate(observation.history, 1):
            print(f"\n--- Step {i} ---")
            print(f"Tool: {entry.get('tool')}")
            print(f"Args: {json.dumps(entry.get('args', {}), indent=2)}")
            print(f"Result:\n{entry.get('result')}")
        print(f"\nGround truth: {ground_truth}")
        print(f"{'='*60}\n")

    return {
        "episode": episode_num,
        "company": observation.company,
        "question": observation.question,
        "submitted_answer": submitted_answer,
        "ground_truth": ground_truth,
        "reward": reward,
        "steps": observation.step_count,
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_api_key(api_base: str) -> str:
    """Get API key based on the API base URL."""
    key = os.getenv("API_KEY")
    if key:
        return key

    if "huggingface" in api_base:
        key = os.getenv("HF_TOKEN")
    elif "openai" in api_base:
        key = os.getenv("OPENAI_API_KEY")
    elif "together" in api_base:
        key = os.getenv("TOGETHER_API_KEY")
    elif "anthropic" in api_base:
        key = os.getenv("ANTHROPIC_API_KEY")
    else:
        # For local servers (vLLM, Ollama), key might not be needed
        key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "not-needed"

    return key or ""


def main():
    parser = argparse.ArgumentParser(
        description="FinQA inference example - works with any OpenAI-compatible API"
    )
    parser.add_argument("--games", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--trace", action="store_true", help="Show full trace (untruncated tool results)")
    parser.add_argument("--server", type=str, default=None, help="FinQA server URL (if already running)")
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_API_BASE})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set via environment variable)"
    )
    args = parser.parse_args()

    global VERBOSE, TRACE, MODEL
    VERBOSE = args.verbose
    TRACE = args.trace
    MODEL = args.model

    # Get API key
    api_key = args.api_key or get_api_key(args.api_base)
    if not api_key:
        print("Error: No API key found")
        print("Set via --api-key or environment variable (HF_TOKEN, OPENAI_API_KEY, etc.)")
        sys.exit(1)

    print("FinQA Inference Example")
    print(f"API Base: {args.api_base}")
    print(f"Model: {MODEL}")
    print(f"Episodes: {args.games}")
    print()

    client = OpenAI(base_url=args.api_base, api_key=api_key)

    if args.server:
        print(f"Connecting to existing server: {args.server}")
        env = FinQAEnv(base_url=args.server)
    else:
        print("Starting FinQA environment from Docker image...")
        try:
            env = FinQAEnv.from_docker_image("finqa-env:latest")
        except RuntimeError as e:
            if "Docker" in str(e):
                print(f"\nError: {e}")
                print("\nOptions:")
                print("  1. Start Docker Desktop and try again")
                print("  2. Start server manually and use --server flag:")
                print("     cd src/envs/finqa_env && uvicorn server.app:app --port 8000")
                print("     python examples/finqa_inference.py --server http://localhost:8000")
                sys.exit(1)
            raise

    results = []

    try:
        for episode_num in range(1, args.games + 1):
            episode_result = play_finqa_episode(env, client, episode_num)
            results.append(episode_result)

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        total_reward = sum(r["reward"] for r in results)
        correct = sum(1 for r in results if r["reward"] > 0)
        avg_steps = sum(r["steps"] for r in results) / len(results)

        print(f"Episodes: {len(results)}")
        print(f"Correct: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
        print(f"Total reward: {total_reward}")
        print(f"Average steps: {avg_steps:.1f}")

        print()
        print("Per-episode results:")
        for r in results:
            status = "CORRECT" if r["reward"] > 0 else "WRONG"
            print(f"  Episode {r['episode']}: {status} (steps={r['steps']}, company={r['company']})")

    finally:
        print()
        print("Closing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
