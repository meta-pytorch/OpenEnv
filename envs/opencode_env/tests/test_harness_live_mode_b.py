# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end Mode B test: real E2B + in-sandbox proxy + real OpenAI."""

from __future__ import annotations

import os

import pytest

from openenv.core.harness import VerifyResult

from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSessionFactory
from opencode_env.sandbox import E2BSandboxBackend

pytestmark = pytest.mark.skipif(
    not (os.environ.get("E2B_API_KEY") and os.environ.get("OPENAI_API_KEY")),
    reason="needs E2B_API_KEY and OPENAI_API_KEY",
)


def _noop_verifier(sandbox, task):
    return VerifyResult(env_reward=0.0, done=True)


def test_mode_b_captures_proxy_trace():
    cfg = OpenCodeConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="openai/gpt-5.3-chat-latest",
        disabled_tools=["webfetch"],
        agent_timeout_s=180.0,
    )
    factory = OpenCodeSessionFactory(
        config=cfg,
        sandbox_backend=E2BSandboxBackend(),
        mode="transparent_proxy",
        verifier=_noop_verifier,
    )

    task = {"instruction": "Print the string 'hello mode b' to stdout and exit."}
    session = factory.create(task=task)
    try:
        exit_code = session.wait_for_completion(timeout_s=150)
        print(f"\nexit={exit_code}")

        # Diagnostics
        for label, path in [
            ("opencode.json", "/home/user/.config/opencode/opencode.json"),
            ("proxy.log", "/home/user/logs/agent/proxy.log"),
            ("proxy_trace", "/home/user/logs/agent/proxy_trace.jsonl"),
            ("opencode.jsonl", "/home/user/logs/agent/opencode.jsonl"),
        ]:
            try:
                content = session.sandbox.read_text(path)
                print(f"\n===== {label} ({len(content)} chars) =====")
                print(content[-2000:])
            except Exception as exc:
                print(f"\n===== {label}: <read failed: {exc}> =====")

        proxy_trace = session.fetch_proxy_trace()
        print(f"\nproxy_trace_len={len(proxy_trace)}")
        assert exit_code == 0, f"opencode exited non-zero: {exit_code}"
        assert len(proxy_trace) >= 1, "proxy captured no turns"
        first = proxy_trace[0]
        assert first["request"]["logprobs"] is True
        assert len(first["completion_tokens"]) > 0
        assert len(first["per_token_logps"]) == len(first["completion_tokens"])
    finally:
        session.close()
