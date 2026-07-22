# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pi environment for OpenEnv.

Harness primitive that drives one rollout of the ``pi`` coding-agent CLI inside
a sandbox. :class:`PiSessionFactory` / :class:`PiSession` / :class:`PiConfig`
mirror the OpenCode primitive; the sandbox backend and interception proxy are
shared with ``opencode_env`` (to be consolidated into ``openenv.core``).

See ``harness.py``.
"""

from opencode_env.sandbox import (
    HFSandboxBackend,
    SandboxBackend,
    SandboxHandle,
)

from .config import PiConfig
from .harness import PiSession, PiSessionFactory
from .task import PiTask

__all__ = [
    # Harness primitive
    "PiConfig",
    "PiSession",
    "PiSessionFactory",
    "PiTask",
    # Sandbox backend (shared with opencode_env)
    "HFSandboxBackend",
    "SandboxBackend",
    "SandboxHandle",
]
