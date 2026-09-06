# SPDX-License-Identifier: BSD-3-Clause

"""openenvd: privileged environment sidecar daemon.

Provides operator-only task supervision related to issue #1053. This is an
opt-in building block inside an existing container, not an implementation
of the proposed agent, grader, or observer policy surfaces.
"""

from openenv.core.openenvd.daemon import create_app, main
from openenv.core.openenvd.isolation import (
    detect_capabilities,
    IsolationCapabilities,
    IsolationError,
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
    "IsolationError",
    "RestartPolicy",
    "Supervisor",
    "SupervisorEvent",
    "TaskSpec",
    "TaskState",
    "TaskStatus",
    "create_app",
    "detect_capabilities",
    "main",
]
