# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

from opencode_env.config import OpenCodeConfig
from opencode_env.opencode_runtime import (
    build_env_vars,
    build_install_cmd,
    build_opencode_json,
    build_run_cmd,
)


def _openai_cfg(**overrides) -> OpenCodeConfig:
    base = dict(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model="openai/gpt-5.3-codex",
    )
    base.update(overrides)
    return OpenCodeConfig(**base)


def test_opencode_json_has_schema_and_provider_block():
    cfg = _openai_cfg()
    doc = json.loads(build_opencode_json(cfg))
    assert doc["$schema"] == "https://opencode.ai/config.json"
    assert doc["model"] == "intercepted/gpt-5.3-codex"
    provider = doc["provider"]["intercepted"]
    assert provider["npm"] == "@ai-sdk/openai"
    assert provider["options"]["baseURL"] == "https://api.openai.com/v1"
    assert provider["options"]["apiKey"] == "sk-test"
    assert provider["options"]["timeout"] == 600_000


def test_opencode_json_disables_tools_by_default():
    cfg = _openai_cfg()
    doc = json.loads(build_opencode_json(cfg))
    assert doc["tools"] == {"webfetch": False, "question": False}


def test_opencode_json_extra_is_deep_merged():
    cfg = _openai_cfg(extra_opencode_json={"theme": "dark", "provider": {"intercepted": {"options": {"custom": 1}}}})
    doc = json.loads(build_opencode_json(cfg))
    assert doc["theme"] == "dark"
    # Deep merge preserves other keys in the nested options block
    options = doc["provider"]["intercepted"]["options"]
    assert options["baseURL"] == "https://api.openai.com/v1"
    assert options["custom"] == 1


def test_install_cmd_pins_version_when_not_latest():
    cfg = _openai_cfg(opencode_version="0.5.3")
    cmd = build_install_cmd(cfg)
    assert "OPENCODE_VERSION=0.5.3" in cmd
    assert "curl -fsSL https://opencode.ai/install | bash" in cmd
    assert "opencode --version" in cmd
    assert "/home/user/.config/opencode" in cmd


def test_install_cmd_respects_sandbox_home():
    cfg = _openai_cfg(sandbox_home="/root")
    cmd = build_install_cmd(cfg)
    assert "/root/.config/opencode" in cmd
    assert "/home/user" not in cmd


def test_install_cmd_omits_version_env_when_latest():
    cfg = _openai_cfg(opencode_version="latest")
    cmd = build_install_cmd(cfg)
    assert "OPENCODE_VERSION" not in cmd


def test_run_cmd_uses_json_format_by_default():
    cfg = _openai_cfg()
    cmd = build_run_cmd(cfg)
    assert "opencode run --format json" in cmd
    assert '"$(cat /home/user/task/instruction.md)"' in cmd
    assert "tee /home/user/logs/agent/opencode.jsonl" in cmd


def test_run_cmd_default_format_has_no_flag():
    cfg = _openai_cfg(run_format="default")
    cmd = build_run_cmd(cfg)
    assert "--format" not in cmd


def test_env_vars_default_to_config_url():
    cfg = _openai_cfg()
    env = build_env_vars(cfg)
    assert env["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["OPENCODE_CONFIG"] == "/home/user/.config/opencode/opencode.json"


def test_env_vars_respect_proxy_override():
    cfg = _openai_cfg(extra_env={"EXTRA": "yes"})
    env = build_env_vars(cfg, base_url_override="http://localhost:7000/v1")
    assert env["OPENAI_BASE_URL"] == "http://localhost:7000/v1"
    assert env["EXTRA"] == "yes"
