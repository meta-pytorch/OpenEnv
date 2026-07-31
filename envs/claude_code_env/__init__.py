# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Claude Code environment for OpenEnv.

Two layers in this package:

1. **Harness primitive** — :class:`ClaudeCodeSessionFactory` / :class:`ClaudeCodeSession` /
   :class:`ClaudeCodeConfig` / :class:`ClaudeCodeTask`. Used in-process to drive one rollout
   inside a sandbox. See ``harness.py``.

2. **Deployable env** — :class:`ClaudeCodeEnv` (MCP client) talks to the FastAPI
   server at ``server/app.py`` over HTTP. Use this when the sandbox + agent
   live behind an HTTP boundary (e.g. an HF Space). See ``client.py`` and
   ``server/``.

The sandbox backend and interception proxy come from ``openenv.core.sandbox``.
"""

from openenv.core.sandbox import (
    HFSandboxBackend,
    SandboxBackend,
    SandboxHandle,
)

from .client import ClaudeCodeEnv
from .config import ClaudeCodeConfig
from .harness import ClaudeCodeSession, ClaudeCodeSessionFactory
from .models import (
    CommandResult,
    ClaudeCodeState,
    RolloutResult,
    RolloutTurn,
)
from .task import ClaudeCodeTask

__all__ = [
    # Deployed-env client
    "ClaudeCodeEnv",
    # HTTP API models
    "CommandResult",
    "ClaudeCodeState",
    "RolloutResult",
    "RolloutTurn",
    # Harness primitive
    "ClaudeCodeConfig",
    "ClaudeCodeSession",
    "ClaudeCodeSessionFactory",
    "ClaudeCodeTask",
    # Sandbox backend (from openenv.core.sandbox)
    "HFSandboxBackend",
    "SandboxBackend",
    "SandboxHandle",
]
