# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Container runtime providers."""

from .providers import (
    ContainerProvider,
    DockerSwarmProvider,
    KubernetesProvider,
    LocalDockerProvider,
    RuntimeProvider,
)
from .uv_provider import UVProvider

# Note: optional cloud providers that require extra SDKs (e.g.
# `ACASandboxProvider`, `DaytonaProvider`) are intentionally NOT re-exported
# here. Import them from their module, e.g.
# `from openenv.core.containers.runtime.aca_provider import ACASandboxProvider`.

__all__ = [
    "ContainerProvider",
    "DockerSwarmProvider",
    "LocalDockerProvider",
    "KubernetesProvider",
    "RuntimeProvider",
    "UVProvider",
]
