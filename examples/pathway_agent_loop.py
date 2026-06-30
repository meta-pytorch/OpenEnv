#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# OpenAI tool-calling agent for pathway_analysis_env (in-process orchestrator).
#
# Usage:
#   export OPENAI_API_KEY=...
#   PYTHONPATH=src:envs uv run python examples/pathway_agent_loop.py \
#       --case toy_case_001.json

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from pathway_analysis_env.agent_openai_tools import (
    OPENAI_TOOLS,
    observation_to_tool_result_content,
    tool_call_to_pathway_action,
)
from pathway_analysis_env.models import PathwayAction
from pathway_analysis_env.server.pathway_environment import PathwayEnvironment


SYSTEM_PROMPT = """You are a computational biologist agent operating a pathway analysis environment.

Required workflow (eval mode):
1. understand_experiment_design and/or inspect_dataset — learn groups and sample layout.
2. run_differential_expression — set reference (baseline) vs alternate (treatment) conditions.
3. run_pathway_enrichment — ORA on DE genes (do not pass a custom gene_list).
4. Optionally compare_pathways between two top pathway names.
5. submit_answer — one pathway hypothesis string supported by ORA.

Rules:
- Never guess without running DE and ORA first.
- Use condition names exactly as returned in available_conditions.
- For submit_answer, name a specific pathway (e.g. from top_pathways), not a long essay.
"""


async def run_episode(
    case_file: str,
    model: str,
    max_turns: int,
    *,
    strict: bool,
) -> dict:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("Install openai: uv add openai") from exc

    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set", file=sys.stderr)

    client = AsyncOpenAI()
    env = PathwayEnvironment(case_file=case_file)
    obs = env.reset(orchestrator_mode=True, strict=strict)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Episode started for case {case_file}. "
                f"Conditions: {obs.available_conditions}. "
                f"{obs.message}"
            ),
        },
    ]

    for turn in range(max_turns):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            if env.state.is_done:
                break
            continue

        messages.append(msg.model_dump())
        for tc in msg.tool_calls:
            action = tool_call_to_pathway_action(
                name=tc.function.name,
                arguments_json=tc.function.arguments,
            )
            step_obs = env.step(action)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation_to_tool_result_content(step_obs),
                }
            )
            if step_obs.done:
                return {
                    "turns": turn + 1,
                    "done": True,
                    "episode_outcome": env.episode_outcome,
                    "last_message": step_obs.message,
                    "steps": env.state.step_count,
                }

    return {
        "turns": max_turns,
        "done": env.state.is_done,
        "episode_outcome": env.episode_outcome,
        "steps": env.state.step_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM agent on pathway_analysis_env")
    parser.add_argument("--case", default="toy_case_001.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        run_episode(args.case, args.model, args.max_turns, strict=args.strict)
    )
    print(json.dumps(result, indent=2))
    outcome = result.get("episode_outcome") or {}
    if outcome.get("correct"):
        sys.exit(0)
    sys.exit(1 if result.get("done") else 2)


if __name__ == "__main__":
    main()
