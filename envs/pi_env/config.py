# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration model for the Pi harness primitive."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PiConfig(BaseModel):
    """All configuration required to launch one Pi rollout in a sandbox.

    Pi (the ``pi`` coding-agent CLI) is pointed at an OpenAI-compatible endpoint
    via a ``models.json`` provider block written under ``PI_CODING_AGENT_DIR``.
    The primitive fills that block from ``base_url`` / ``api_key`` / ``model``
    and runs ``pi --print`` headless.
    """

    # --- LLM endpoint ---------------------------------------------------------
    base_url: str
    api_key: str = "intercepted"
    model: str = "intercepted-model"

    # --- Pi CLI ---------------------------------------------------------------
    pi_version: str = "latest"
    # Provider name registered in models.json and selected via ``--provider``.
    provider_name: str = "intercepted"
    system_prompt: str | None = None
    # Restrict Pi's tools to this allowlist (passed to ``--tools``). ``None``
    # keeps all builtin tools (read/bash/edit/write/grep/find/ls) enabled.
    tools: list[str] | None = None
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
    # used in ``mode="transparent_proxy"``. ``None`` disables the cap.
    proxy_max_tokens_cap: int | None = 16_384
    # Per-turn top-k logprobs the proxy requests from the upstream.
    proxy_top_logprobs: int = 5
    # Disable reasoning/thinking mode for Qwen3 / Qwen3.5 models. Proxy sets
    # ``extra_body.chat_template_kwargs.enable_thinking=false`` on forwarded
    # requests. Ignored by providers that don't support the field.
    proxy_disable_thinking: bool = False
