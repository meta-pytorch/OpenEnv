# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure builders for Claude Code sandbox bootstrap artifacts.

These functions produce the exact files and shell commands the sandbox needs to
run Claude Code against a configured LLM endpoint. No IO, no sandbox coupling —
the sandbox backend is responsible for writing files and running commands.

Claude Code speaks the Anthropic Messages API only, so it is pointed (via
``ANTHROPIC_BASE_URL``) at the in-sandbox translation shim rather than at the
OpenAI-compatible endpoint directly. The shim, the interception proxy and vLLM
are wired up by the harness; this module only knows the shim's URL.
"""

from __future__ import annotations

import json

from .config import ClaudeCodeConfig

# Claude Code requires Node >= 22; bootstrap this version when the image ships none.
_NODE_VERSION = "22.19.0"
# npm package for the Claude Code CLI.
_CLAUDE_NPM_PACKAGE = "@anthropic-ai/claude-code"


def claude_config_dir(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/.claude"


def settings_path(config: ClaudeCodeConfig) -> str:
    return f"{claude_config_dir(config)}/settings.json"


def instruction_path(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/task/instruction.md"


def system_prompt_path(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/task/system.md"


def agent_log_path(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/claude.jsonl"


def verifier_reward_path(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/verifier/reward.txt"


def workdir_path(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/workdir"


def node_dir(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/.node"


def npm_prefix(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/.claude-npm"


def claude_bin_path(config: ClaudeCodeConfig) -> str:
    return f"{npm_prefix(config)}/bin/claude"


def proxy_dir(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/proxy"


def proxy_source_path(config: ClaudeCodeConfig) -> str:
    return f"{proxy_dir(config)}/interception.py"


def proxy_trace_path(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/proxy_trace.jsonl"


def proxy_log_path(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/proxy.log"


def shim_dir(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/shim"


def shim_source_path(config: ClaudeCodeConfig) -> str:
    return f"{shim_dir(config)}/anthropic_shim.py"


def shim_log_path(config: ClaudeCodeConfig) -> str:
    return f"{config.sandbox_home}/logs/agent/shim.log"


def build_settings(config: ClaudeCodeConfig) -> str:
    """Return the serialized ``.claude/settings.json``.

    Marks onboarding complete so ``claude --print`` does not block on first-run
    theme/login prompts, and turns off the background auto-updater.
    """
    doc = {
        "hasCompletedOnboarding": True,
        "autoUpdaterStatus": "disabled",
    }
    return json.dumps(doc, indent=2)


def build_install_cmd(config: ClaudeCodeConfig) -> str:
    """Return the shell command that installs Claude Code (bootstrapping Node if needed)."""
    home = config.sandbox_home
    package = _CLAUDE_NPM_PACKAGE
    if config.claude_code_version and config.claude_code_version != "latest":
        package = f"{package}@{config.claude_code_version}"
    return (
        "set -e && "
        f"mkdir -p {home}/.claude {home}/logs/agent {home}/logs/verifier {home}/task {home}/workdir && "
        # Bootstrap Node 22 when the image ships none new enough (Claude Code needs >=22).
        'if ! command -v node >/dev/null 2>&1 || '
        '! node -e "const v=process.versions.node.split(\'.\').map(Number); '
        'process.exit(v[0]>22||(v[0]===22&&v[1]>=19)?0:1)"; then '
        "A=$(uname -m); case $A in x86_64) A=x64;; aarch64|arm64) A=arm64;; esac && "
        f"curl -fsSL https://nodejs.org/dist/v{_NODE_VERSION}/node-v{_NODE_VERSION}-linux-$A.tar.xz "
        f"| tar -xJ -C {home} && ln -sfn {home}/node-v{_NODE_VERSION}-linux-$A {node_dir(config)}; fi && "
        f'export PATH="{node_dir(config)}/bin:$PATH" && '
        f"npm install -g --prefix {npm_prefix(config)} {package} && "
        f"{claude_bin_path(config)} --version"
    )


def build_run_cmd(config: ClaudeCodeConfig) -> str:
    """Return the shell command that launches Claude Code headless against a task."""
    tools_flag = f"--allowedTools {','.join(config.tools)} " if config.tools else ""
    turns_flag = f"--max-turns {config.max_turns} " if config.max_turns else ""
    system_flag = (
        f'--append-system-prompt "$(cat {system_prompt_path(config)})" '
        if config.system_prompt
        else ""
    )
    return (
        "set -o pipefail && "
        f'export PATH="{node_dir(config)}/bin:{npm_prefix(config)}/bin:$PATH" && '
        f"cd {workdir_path(config)} && "
        f"{claude_bin_path(config)} -p --output-format json --dangerously-skip-permissions "
        f"--model {config.model} "
        f"{turns_flag}{tools_flag}{system_flag}"
        f'"$(cat {instruction_path(config)})" '
        f"2>&1 | tee {agent_log_path(config)}"
    ).strip()


def build_env_vars(config: ClaudeCodeConfig, *, anthropic_base_url: str) -> dict[str, str]:
    """Return env vars to set on the Claude Code process.

    ``anthropic_base_url`` is the endpoint Claude Code talks to: the in-sandbox
    shim in ``transparent_proxy`` mode, or ``config.base_url`` directly in
    ``black_box`` mode. ``DISABLE_NON_ESSENTIAL_MODEL_CALLS`` keeps title/flavor
    generation from polluting the captured trace. ``IS_SANDBOX`` lets
    ``--dangerously-skip-permissions`` run as root (the sandbox user).
    """
    env = dict(config.extra_env)
    env["ANTHROPIC_BASE_URL"] = anthropic_base_url
    env["ANTHROPIC_API_KEY"] = config.api_key
    env["CLAUDE_CONFIG_DIR"] = claude_config_dir(config)
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["DISABLE_NON_ESSENTIAL_MODEL_CALLS"] = "1"
    env["IS_SANDBOX"] = "1"
    return env
