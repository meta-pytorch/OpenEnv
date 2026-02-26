# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""OpenClaw agent type plugin.

Self-contained plugin: SpawnInfo dataclass + Plugin class + binary resolution.
OpenClaw agents are TypeScript/Node.js-based, backed by openclaw-agent-server.
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.config import Agent, SpawnRequest


@dataclass
class OpenClawSpawnInfo:
    """OpenClaw-specific spawn configuration.

    Packed into SpawnRequest.spawn_info / Agent.spawn_info.
    """

    system_prompt: str = ""
    model: str = "claude-sonnet-4-5"
    provider: str = "anthropic"
    tools: list[str] = field(default_factory=lambda: ["bash"])
    thinking_level: str = "none"  # "none", "low", "medium", "high"
    api_key: str = ""
    base_url: str = ""


@dataclass
class OpenClawControlRequest:
    pass


@dataclass
class OpenClawControlResponse:
    pass


class OpenClawPlugin:
    """AgentTypePlugin implementation for OpenClaw agents."""

    @property
    def agent_type(self) -> str:
        return "openclaw"

    def build_config(
        self,
        agent: Agent,
        nonce: str,
        http_port: int,
        workspace_path: Path,
        request: SpawnRequest,
    ) -> dict[str, Any]:
        if not isinstance(agent.spawn_info, OpenClawSpawnInfo):
            raise TypeError(
                f"Expected OpenClawSpawnInfo, got {type(agent.spawn_info).__name__}"
            )
        info = agent.spawn_info
        config: dict[str, Any] = {
            "agent_id": agent.id,
            "nonce": nonce,
            "name": agent.name,
            "agent_type": "openclaw",
            "system_prompt": info.system_prompt,
            "tools": info.tools,
            "model": info.model,
            "provider": info.provider,
            "thinking_level": info.thinking_level,
            "http_port": http_port,
        }
        if info.api_key:
            config["api_key"] = info.api_key
        if info.base_url:
            config["base_url"] = info.base_url
        return config

    def resolve_command(self) -> list[str]:
        bin_path = os.environ.get("OPENCLAW_RUNNER_BIN") or shutil.which(
            "openclaw-agent-server"
        )
        if not bin_path:
            raise FileNotFoundError(
                "openclaw-agent-server binary not found. "
                "Set OPENCLAW_RUNNER_BIN or add it to PATH. "
                "Build with: cd agentkernel/openclaw_runner && pnpm install && pnpm build"
            )
        return [bin_path]
