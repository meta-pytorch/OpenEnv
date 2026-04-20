# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end Mode B test against a live vLLM + cloudflared tunnel.

Unlike the OpenAI smoke (which refuses logprobs on gpt-5.x), vLLM grants
logprobs freely. This is the test that validates the full training-time
path: E2B sandbox → in-sandbox proxy → cloudflare tunnel → vLLM.

Run::

    VLLM_TUNNEL_URL=https://<...>.trycloudflare.com/v1 \\
    VLLM_MODEL=Qwen/Qwen3.5-4B \\
    E2B_API_KEY=... \\
    PYTHONPATH=src:envs/opencode_env uv run pytest \\
    envs/opencode_env/tests/test_harness_live_vllm.py -v -s
"""

from __future__ import annotations

import os

import pytest

from openenv.core.harness import VerifyResult

from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.live_watch import collect_rollout_summary, print_rollout_summary
from opencode_env.sandbox import E2BSandboxBackend

pytestmark = pytest.mark.skipif(
    not (os.environ.get("E2B_API_KEY") and os.environ.get("VLLM_TUNNEL_URL")),
    reason="needs E2B_API_KEY and VLLM_TUNNEL_URL",
)


def _fizzbuzz_verifier(sandbox, task):
    r = sandbox.exec(
        "cd /home/user/workdir && python fizzbuzz.py 2>&1 | head -20",
        timeout=30,
    )
    stdout = r.stdout or ""
    expected_bits = ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7"]
    hits = sum(1 for tok in expected_bits if tok in stdout)
    return VerifyResult(
        env_reward=float(hits / len(expected_bits)),
        done=True,
        metrics={"hits": hits, "total": len(expected_bits)},
        artifacts={"stdout": stdout[:2000]},
    )


def test_mode_b_against_vllm_tunnel():
    tunnel = os.environ["VLLM_TUNNEL_URL"].rstrip("/")
    if not tunnel.endswith("/v1"):
        tunnel = tunnel + "/v1"
    model_name = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-4B")

    cfg = OpenCodeConfig(
        provider="openai_compatible",
        base_url=tunnel,
        api_key="intercepted",  # vLLM ignores this
        model=f"openai_compatible/{model_name}",
        disabled_tools=["webfetch"],
        agent_timeout_s=600.0,
        proxy_max_tokens_cap=4096,
        proxy_disable_thinking=True,  # Qwen3.5 emits lots of thinking otherwise
    )
    factory = OpenCodeSessionFactory(
        config=cfg,
        sandbox_backend=E2BSandboxBackend(),
        mode="transparent_proxy",
        verifier=_fizzbuzz_verifier,
    )

    task = {
        "instruction": (
            "Write a Python script `fizzbuzz.py` in the current directory "
            "that prints fizzbuzz for numbers 1..15, one per line. "
            "Use 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, "
            "'FizzBuzz' for both."
        ),
    }
    session = factory.create(task=task)
    print(f"sandbox_id={session.sandbox.sandbox_id}", flush=True)
    try:
        exit_code = session.wait_for_completion(timeout_s=540)
        print(f"\nopencode exit code: {exit_code}")

        # Post-rollout summary — reads logs + workdir, prints a structured report
        summary = collect_rollout_summary(session)
        print_rollout_summary(summary)

        result = session.verify(transcript=[])
        print(
            f"\nverifier reward: {result.env_reward}  metrics: {result.metrics}"
        )
        print(f"verifier stdout: {result.artifacts.get('stdout', '')[:400]}")

        # Assertions
        productive = [t for t in summary.proxy_turns if t["completion_tokens"]]
        assert exit_code == 0, f"opencode exited non-zero: {exit_code}"
        assert summary.proxy_turn_count() >= 1, "proxy captured no turns"
        assert len(productive) >= 1, (
            "proxy captured no turns with completion tokens — "
            "logprob capture path is broken"
        )
        first = productive[0]
        assert first["request"].get("logprobs") is True
        assert len(first["per_token_logps"]) == len(first["completion_tokens"])
        assert result.env_reward is not None
        assert result.env_reward >= 0.5, (
            f"reward {result.env_reward} too low; opencode did not produce a "
            "working fizzbuzz."
        )
    finally:
        session.close()
