# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core components for agentic environments."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from . import env_server
from .env_server import *  # noqa: F403

if TYPE_CHECKING:
    from .env_client import EnvClient
    from .generic_client import GenericAction, GenericEnvClient
    from .harness import (
        CLIHarnessAdapter,
        HarnessAdapter,
        HarnessEvent,
        HarnessRolloutResult,
        HarnessRunLimits,
        MCPHarnessAdapter,
        ModelStep,
        ModelStepResult,
        ResourceSession,
        ResourceSessionFactory,
        SessionMCPBridge,
        StepEnvSessionAdapter,
        ToolResult,
        VerifyResult,
        build_harness_rollout_func,
    )
    from .llm_client import (
        AnthropicClient,
        create_llm_client,
        LLMClient,
        LLMResponse,
        OpenAIClient,
        ToolCall,
    )
    from .mcp_client import MCPClientBase, MCPToolClient
    from .sync_client import SyncEnvClient

__all__ = [
    "EnvClient",
    "SyncEnvClient",
    "GenericEnvClient",
    "GenericAction",
    "MCPClientBase",
    "MCPToolClient",
    "ResourceSession",
    "ResourceSessionFactory",
    "HarnessAdapter",
    "HarnessRunLimits",
    "HarnessEvent",
    "HarnessRolloutResult",
    "ModelStep",
    "ModelStepResult",
    "ToolResult",
    "VerifyResult",
    "StepEnvSessionAdapter",
    "SessionMCPBridge",
    "MCPHarnessAdapter",
    "CLIHarnessAdapter",
    "build_harness_rollout_func",
    "AnthropicClient",
    "LLMClient",
    "LLMResponse",
    "OpenAIClient",
    "ToolCall",
    "create_llm_client",
] + env_server.__all__  # type: ignore


_LAZY_ATTRS = {
    "EnvClient": (".env_client", "EnvClient"),
    "SyncEnvClient": (".sync_client", "SyncEnvClient"),
    "GenericEnvClient": (".generic_client", "GenericEnvClient"),
    "GenericAction": (".generic_client", "GenericAction"),
    "MCPClientBase": (".mcp_client", "MCPClientBase"),
    "MCPToolClient": (".mcp_client", "MCPToolClient"),
    "ResourceSession": (".harness", "ResourceSession"),
    "ResourceSessionFactory": (".harness", "ResourceSessionFactory"),
    "HarnessAdapter": (".harness", "HarnessAdapter"),
    "HarnessRunLimits": (".harness", "HarnessRunLimits"),
    "HarnessEvent": (".harness", "HarnessEvent"),
    "HarnessRolloutResult": (".harness", "HarnessRolloutResult"),
    "ModelStep": (".harness", "ModelStep"),
    "ModelStepResult": (".harness", "ModelStepResult"),
    "ToolResult": (".harness", "ToolResult"),
    "VerifyResult": (".harness", "VerifyResult"),
    "StepEnvSessionAdapter": (".harness", "StepEnvSessionAdapter"),
    "SessionMCPBridge": (".harness", "SessionMCPBridge"),
    "MCPHarnessAdapter": (".harness", "MCPHarnessAdapter"),
    "CLIHarnessAdapter": (".harness", "CLIHarnessAdapter"),
    "build_harness_rollout_func": (".harness", "build_harness_rollout_func"),
    "AnthropicClient": (".llm_client", "AnthropicClient"),
    "LLMClient": (".llm_client", "LLMClient"),
    "LLMResponse": (".llm_client", "LLMResponse"),
    "OpenAIClient": (".llm_client", "OpenAIClient"),
    "ToolCall": (".llm_client", "ToolCall"),
    "create_llm_client": (".llm_client", "create_llm_client"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    try:
        value = getattr(env_server, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
