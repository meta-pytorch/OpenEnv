# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sandbox backends for the OpenCode harness.

The primitive ships with :class:`E2BSandboxBackend` as the default; any backend
that satisfies the :class:`SandboxBackend` / :class:`SandboxHandle` protocols
can be swapped in.
"""

from .base import BgJob, ExecResult, SandboxBackend, SandboxHandle
from .e2b import E2BBgJob, E2BSandboxBackend, E2BSandboxHandle

__all__ = [
    "BgJob",
    "ExecResult",
    "SandboxBackend",
    "SandboxHandle",
    "E2BBgJob",
    "E2BSandboxBackend",
    "E2BSandboxHandle",
]
