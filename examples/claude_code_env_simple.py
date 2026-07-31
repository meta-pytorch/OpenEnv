#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end claude_code_env example: write binary_search.py and verify it.

Hits the deployed HF Space ``<user>/claude-code-env`` (override via
``CLAUDE_CODE_ENV_SPACE`` env var to point at your own Space or a local
container). The single MCP tool ``run_rollout`` does:

  1. Spawns a fresh Hugging Face sandbox (using the default image, which
     cold-installs Node + Claude Code per rollout; pass ``image=`` to reuse
     a prebaked image).
  2. Bootstraps an in-sandbox FastAPI proxy that captures per-token
     logprobs (``mode="transparent_proxy"``). An Anthropic-to-OpenAI shim
     sits in front of it so Claude Code (Anthropic Messages API) still
     records the same OpenAI-native capture.
  3. Runs ``claude`` with the instruction.
  4. Executes the verify bash commands; reward = passed / total.
  5. Returns a ``RolloutResult`` with reward + per-turn logprobs +
     the file contents the agent produced.

Prerequisites
-------------
- ``OPENAI_API_KEY`` in the environment (passed to the Space per-call;
  doesn't need to be a Space secret). Swap to ``endpoint="vllm"`` or
  ``endpoint="hf_router"`` for those backends.

Usage::

    PYTHONPATH=src:envs uv run python examples/claude_code_env_simple.py

Expected output (a few minutes with a cold sandbox)::

    reward: 1.0
    turns:  3
    files:  ['/root/workdir/binary_search.py', ...]
    wall:   120.0 s
"""

from __future__ import annotations

import asyncio
import os
import sys

# Make ``envs/`` importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))

from claude_code_env import ClaudeCodeEnv  # noqa: E402
from claude_code_env.client import _extract_text  # noqa: E402
from claude_code_env.models import RolloutResult  # noqa: E402


SPACE = os.environ.get("CLAUDE_CODE_ENV_SPACE", "https://<user>-claude-code-env.hf.space")

INSTRUCTION = (
    "Create a single Python file named `binary_search.py` in the current "
    "working directory. Use the relative path `binary_search.py`. Expose "
    "exactly one function:\n"
    "    def binary_search(arr: list[int], target: int) -> int\n"
    "Return the index of `target` in the sorted list `arr`, or -1 if absent. "
    "Use the binary-search algorithm; do not call list.index."
)

VERIFY = [
    "test -f /root/workdir/binary_search.py",
    "python -c \"import sys; sys.path.insert(0, '/root/workdir'); "
    "import binary_search; "
    "assert binary_search.binary_search([1,2,3,4,5], 3) == 2; "
    "assert binary_search.binary_search([1,2,3], 99) == -1; "
    "assert binary_search.binary_search([], 1) == -1; "
    "print('OK')\"",
]


async def main() -> int:
    endpoint = os.environ.get("CLAUDE_CODE_ENV_ENDPOINT", "openai")  # openai | hf_router | vllm
    model = os.environ.get("CLAUDE_CODE_ENV_MODEL", "gpt-4o-mini")
    # openai needs a key passed per call. hf_router / vllm resolve theirs from the
    # Space's own secrets (HF_ROUTER_API_KEY / VLLM_API_KEY), so leave api_key unset.
    api_key = os.environ.get("OPENAI_API_KEY")
    if endpoint == "openai" and not api_key:
        print(
            "ERROR: endpoint=openai needs OPENAI_API_KEY. Set CLAUDE_CODE_ENV_ENDPOINT=hf_router "
            "(open model via Inference Providers, uses the Space's HF_ROUTER_API_KEY secret) to run "
            "without an OpenAI key.",
            file=sys.stderr,
        )
        return 2
    if endpoint == "hf_router" and model == "gpt-4o-mini":
        model = "Qwen/Qwen3-4B-Instruct-2507:nscale"  # open model default for the router

    print(f"Hitting Space:   {SPACE}")
    print(f"Endpoint:        {endpoint} ({model})")
    print(f"Instruction:     {INSTRUCTION.splitlines()[0]} ...")
    print()

    rollout_kwargs = dict(
        endpoint=endpoint,  # vllm | openai | hf_router
        model=model,
        instruction=INSTRUCTION,
        setup=[],  # no setup commands
        verify=VERIFY,
        task_id="binary_search_simple",
        image="",  # default sandbox image (cold-installs Node + Claude Code)
        agent_timeout_s=600,
    )
    if api_key:
        rollout_kwargs["api_key"] = api_key  # openai; hf_router/vllm use the Space secret

    env = ClaudeCodeEnv(base_url=SPACE)
    env.use_production_mode = True
    try:
        raw = await env.call_tool("run_rollout", **rollout_kwargs)
        result = RolloutResult.model_validate_json(_extract_text(raw))
    finally:
        await env.close()

    print("--- result ---")
    print(f"reward:    {result.reward}")
    print(f"turns:     {len(result.proxy_turns)}")
    print(f"tokens:    {sum(len(t.completion_tokens) for t in result.proxy_turns)}")
    print(f"sandbox:   {result.sandbox_id}")
    print(f"wall_s:    {result.wall_s}")
    print(f"files:     {sorted(result.files)}")
    print(f"verify:    {[(v.cmd[:40], v.exit_code) for v in result.verify_results]}")
    if result.error:
        print(f"error:     {result.error}")

    if result.proxy_turns:
        first = next((t for t in result.proxy_turns if t.completion_tokens), None)
        if first:
            print()
            print("--- first productive turn (first 8 tokens with logprobs) ---")
            toks = first.completion_tokens[:8]
            lps = first.per_token_logps[:8]
            for tok, lp in zip(toks, lps):
                print(f"  {tok!r:<14}  {lp:+.3f}")

    return 0 if result.reward == 1.0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
