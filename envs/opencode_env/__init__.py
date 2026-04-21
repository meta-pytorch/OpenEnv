# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenCode harness primitive for OpenEnv.

A reusable building block that, given an OpenAI-compatible base URL and a
sandbox backend, runs an autonomous OpenCode agent to completion in an isolated
sandbox and exposes every LLM turn (messages, tools, response, optionally
logprobs) to the caller.

See ``envs/opencode_env/README.md`` for usage.
"""

from .config import OpenCodeConfig, Provider
from .harness import OpenCodeSession, OpenCodeSessionFactory
from .live_watch import RolloutSummary, collect_rollout_summary, print_rollout_summary
from .sandbox import E2BSandboxBackend, SandboxBackend, SandboxHandle
from .task import OpenCodeTask

__all__ = [
    "OpenCodeConfig",
    "OpenCodeSession",
    "OpenCodeSessionFactory",
    "OpenCodeTask",
    "E2BSandboxBackend",
    "Provider",
    "RolloutSummary",
    "SandboxBackend",
    "SandboxHandle",
    "collect_rollout_summary",
    "print_rollout_summary",
]
