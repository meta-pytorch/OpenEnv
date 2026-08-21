# SPDX-License-Identifier: BSD-3-Clause

"""Task specification and status models for openenvd."""

from __future__ import annotations

import enum
import time
from typing import Optional

from pydantic import BaseModel, Field


class RestartPolicy(str, enum.Enum):
    """When openenvd restarts an exited task.

    - ``NEVER``: leave the task dead after its first exit.
    - ``ON_FAILURE``: restart only after a nonzero exit, up to ``max_retries``
      times with exponential backoff.
    - ``ALWAYS``: restart unconditionally until explicitly stopped.
    """

    NEVER = "never"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"


class TaskState(str, enum.Enum):
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    RESTARTING = "restarting"
    EXITED = "exited"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskSpec(BaseModel):
    """Declarative description of a sidecar task supervised by openenvd.

    Attributes:
        name (`str`): Unique task name (lowercase alphanumerics, ``-``, ``_``).
        argv (`list[str]`): Command to execute.
        env (`dict[str, str]`): Extra environment variables merged over the
            daemon's own environment.
        cwd (`Optional[str]`): Working directory for the child.
        uid (`Optional[int]`): Run as this UID (overrides ``auto_uid``).
        gid (`Optional[int]`): Run as this GID (overrides ``auto_uid``).
        auto_uid (`bool`): When no explicit uid/gid is given and the daemon
            is privileged, allocate a dedicated UID pair for this task so
            sibling tasks cannot signal it or read its files.
        network_isolated (`bool`): Run the child in its own network namespace
            when the runtime allows it; falls back to the shared namespace
            with a warning otherwise.
        restart_policy (`RestartPolicy`): Restart behavior on exit.
        max_retries (`int`): Restart budget for ``ON_FAILURE``.
        backoff_s (`float`): Base delay between restarts (exponential).
        stop_grace_s (`float`): Seconds between SIGTERM and SIGKILL on stop.
    """

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    argv: list[str] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    uid: Optional[int] = Field(default=None, ge=0)
    gid: Optional[int] = Field(default=None, ge=0)
    auto_uid: bool = True
    network_isolated: bool = False
    restart_policy: RestartPolicy = RestartPolicy.NEVER
    max_retries: int = Field(default=3, ge=0)
    backoff_s: float = Field(default=0.5, gt=0)
    stop_grace_s: float = Field(default=5.0, ge=0)


class TaskStatus(BaseModel):
    """Point-in-time snapshot of a task's lifecycle state."""

    name: str
    state: TaskState
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    restarts: int = 0
    uid: Optional[int] = None
    gid: Optional[int] = None


class SupervisorEvent(BaseModel):
    """A lifecycle event emitted by the supervisor.

    ``seq`` is a daemon-monotonic sequence number; consumers use it to resume
    an event stream without gaps or duplicates.
    """

    seq: int = 0
    ts: float = Field(default_factory=time.time)
    task: str
    kind: str
    detail: Optional[str] = None
