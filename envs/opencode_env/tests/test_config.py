# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from opencode_env.config import OpenCodeConfig, provider_npm_package


def test_defaults_require_only_base_url():
    cfg = OpenCodeConfig(base_url="http://localhost:8000/v1")
    assert cfg.provider == "openai_compatible"
    assert cfg.api_key == "intercepted"
    assert cfg.model == "intercepted/model"
    assert cfg.opencode_version == "latest"
    assert "webfetch" in cfg.disabled_tools
    assert cfg.run_format == "json"


def test_provider_npm_mapping():
    assert provider_npm_package("openai_compatible") == "@ai-sdk/openai-compatible"
    assert provider_npm_package("openai") == "@ai-sdk/openai"
    assert provider_npm_package("anthropic") == "@ai-sdk/anthropic"


def test_rejects_unknown_provider():
    with pytest.raises(ValueError):
        OpenCodeConfig(provider="bogus", base_url="x")  # type: ignore[arg-type]


def test_custom_fields_override_defaults():
    cfg = OpenCodeConfig(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model="openai/gpt-5.3-codex",
        opencode_version="0.5.3",
        disabled_tools=["webfetch"],
        system_prompt="be brief",
        extra_env={"FOO": "bar"},
    )
    assert cfg.model == "openai/gpt-5.3-codex"
    assert cfg.opencode_version == "0.5.3"
    assert cfg.extra_env == {"FOO": "bar"}
