# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end Mode A test: real E2B sandbox + real OpenAI API.

Skipped unless both ``E2B_API_KEY`` and ``OPENAI_API_KEY`` are set. Verifies
that the full factory.create(task) → opencode runs → session.verify path
works against a real provider.

Run with::

    E2B_API_KEY=... OPENAI_API_KEY=... \\
    PYTHONPATH=src:envs/opencode_env uv run pytest \\
    envs/opencode_env/tests/test_harness_live_openai.py -v -s
"""

from __future__ import annotations

import os

import pytest

from openenv.core.harness import VerifyResult

from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.opencode_runtime import workdir_path
from opencode_env.sandbox import E2BSandboxBackend

pytestmark = pytest.mark.skipif(
    not (os.environ.get("E2B_API_KEY") and os.environ.get("OPENAI_API_KEY")),
    reason="needs E2B_API_KEY and OPENAI_API_KEY",
)


def _fizzbuzz_verifier(sandbox, task):
    """Check that fizzbuzz.py exists and produces correct output."""
    home = "/home/user"
    r = sandbox.exec(
        f"cd {home}/workdir && "
        "python fizzbuzz.py 2>&1 | head -20",
        timeout=30,
    )
    stdout = r.stdout or ""
    # Cheap correctness check for the first few values
    expected_bits = ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7"]
    hits = sum(1 for token in expected_bits if token in stdout)
    reward = hits / len(expected_bits)
    return VerifyResult(
        env_reward=float(reward),
        done=True,
        metrics={"hits": hits, "total": len(expected_bits)},
        artifacts={"stdout": stdout},
    )


def test_mode_a_against_openai():
    cfg = OpenCodeConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="openai/gpt-4o-mini",  # cheap, reliable tool-calling
        disabled_tools=["webfetch"],
        agent_timeout_s=300.0,
    )
    factory = OpenCodeSessionFactory(
        config=cfg,
        sandbox_backend=E2BSandboxBackend(),
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
    try:
        exit_code = session.wait_for_completion(timeout_s=240)
        print(f"\nopencode exit code: {exit_code}")
        trace = session.fetch_trace()
        print(f"trace (last 1000 chars):\n{trace[-1000:]}")
        result = session.verify(transcript=[])
        print(f"reward: {result.env_reward}")
        print(f"metrics: {result.metrics}")
        print(f"stdout: {result.artifacts.get('stdout', '')[:500]}")
        # Minimum success bar: opencode ran, exited cleanly, verifier saw *something*
        assert exit_code == 0, f"opencode did not exit cleanly: {exit_code}"
        assert result.env_reward is not None
        assert result.env_reward >= 0.5, (
            f"reward {result.env_reward} too low — opencode may have failed to "
            "create a working fizzbuzz.py"
        )
    finally:
        session.close()
