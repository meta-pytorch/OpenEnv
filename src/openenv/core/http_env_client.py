# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HTTP Environment Client.

HTTPEnvClient is a wrapper around EnvClient that uses the standard State type
and only requires 2 type parameters (ActT, ObsT) instead of 3.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from .env_client import EnvClient
from .env_server.types import State

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")


class HTTPEnvClient(EnvClient[ActT, ObsT, State], Generic[ActT, ObsT]):
    """
    HTTP Environment Client.

    This is a convenience wrapper around EnvClient that uses the standard State type
    and only requires 2 type parameters (action and observation types).

    EnvClient already supports HTTP URLs by converting them to WebSocket connections
    internally, so HTTPEnvClient can use EnvClient's implementation.
    """

    pass


__all__ = ["HTTPEnvClient"]
