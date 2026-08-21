# SPDX-License-Identifier: BSD-3-Clause

"""openenvd: privileged environment sidecar daemon.

Implements the supervision and isolation core of RFC 1053 (policy-scoped
surfaces). openenvd runs as the parent of the environment workload inside
the container, spawns de-privileged children, and exposes a control API for
registering auxiliary sidecar tasks (observers, trajectory writers, agent
buses, ...).
"""

from openenv.core.openenvd.daemon import create_app, main
from openenv.core.openenvd.isolation import (
    build_preexec,
    detect_capabilities,
    IsolationCapabilities,
)
from openenv.core.openenvd.models import (
    RestartPolicy,
    SupervisorEvent,
    TaskSpec,
    TaskState,
    TaskStatus,
)
from openenv.core.openenvd.supervisor import Supervisor

__all__ = [
    "IsolationCapabilities",
    "RestartPolicy",
    "Supervisor",
    "SupervisorEvent",
    "TaskSpec",
    "TaskState",
    "TaskStatus",
    "build_preexec",
    "create_app",
    "detect_capabilities",
    "main",
]
