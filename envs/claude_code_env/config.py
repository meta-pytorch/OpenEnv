# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration model for the Claude Code harness primitive."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClaudeCodeConfig(BaseModel):
    """All configuration required to launch one Claude Code rollout in a sandbox.

    Claude Code speaks the Anthropic Messages API only, so it is pointed at an
    in-sandbox translation shim via ``ANTHROPIC_BASE_URL``. The shim converts
    Anthropic Messages to OpenAI chat-completions, forwards to the interception
    proxy (which captures per-token logprobs against the real vLLM upstream at
    ``base_url``), and translates the reply back. The primitive runs
    ``claude --print`` headless against the task instruction.
    """

    # --- LLM endpoint (the real upstream the proxy forwards to) ---------------
    base_url: str
    api_key: str = "intercepted"
    model: str = "intercepted-model"

    # --- Claude Code CLI ------------------------------------------------------
    claude_code_version: str = "latest"
    system_prompt: str | None = None
    # Restrict Claude Code's tools to this allowlist (passed to ``--allowedTools``).
    # ``None`` keeps all builtin tools (Bash/Read/Edit/Write/Grep/Glob) enabled.
    tools: list[str] | None = None
    # Cap the number of agentic turns (``--max-turns``). ``None`` leaves it unbounded.
    max_turns: int | None = None
    context_window: int = 32_768
    max_tokens: int = 4_096

    # --- CLI invocation -------------------------------------------------------
    agent_timeout_s: float = 900.0
    extra_env: dict[str, str] = Field(default_factory=dict)
    extra_setup_shell: str | None = None

    # --- Sandbox paths --------------------------------------------------------
    # Root directory inside the sandbox where the primitive writes config, task
    # files, and logs. Defaults to ``/root`` (the HF sandbox user); override for
    # a backend whose home differs.
    sandbox_home: str = "/root"

    # --- Transparent-proxy tuning --------------------------------------------
    # Cap ``max_tokens`` / ``max_completion_tokens`` on forwarded requests. Only
    # used in ``mode="transparent_proxy"``. ``None`` disables the cap. Kept well
    # below the vLLM context window because Claude Code's system prompt is large
    # and grows per turn, so ``prompt_tokens + max_tokens`` can otherwise exceed
    # ``max_model_len`` and the upstream returns 400.
    proxy_max_tokens_cap: int | None = 8_192
    # Per-turn top-k logprobs the proxy requests from the upstream.
    proxy_top_logprobs: int = 5
    # Disable reasoning/thinking mode for Qwen3 / Qwen3.5 models. Proxy sets
    # ``extra_body.chat_template_kwargs.enable_thinking=false`` on forwarded
    # requests. Ignored by providers that don't support the field.
    proxy_disable_thinking: bool = False
