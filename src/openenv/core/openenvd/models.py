# SPDX-License-Identifier: BSD-3-Clause

"""Task specification and status models for openenvd."""

from __future__ import annotations

import enum
import time
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
        env (`dict[str, str]`): Explicit child environment. Only a standard
            executable search path is supplied by default; daemon secrets
            and interpreter settings are never inherited.
        cwd (`str`, *optional*): Child working directory; defaults to ``/``.
        uid (`int`, *optional*): Non-root UID; must be supplied with ``gid``.
        gid (`int`, *optional*): Non-root GID; must be supplied with ``uid``.
        auto_uid (`bool`): When no explicit uid/gid is given and the daemon
            is privileged, allocate a dedicated UID pair. Registration fails
            if that is unavailable. Set to ``False`` only for trusted tasks
            that may run as the daemon user.
        network_isolated (`bool`): Require a separate Linux network namespace
            and an unprivileged UID/GID. Never falls back to shared networking.
        restart_policy (`RestartPolicy`): Restart behavior on exit.
        max_retries (`int`): Restart budget for ``ON_FAILURE``.
        backoff_s (`float`): Base delay between restarts (exponential).
        stop_grace_s (`float`): Seconds between SIGTERM and SIGKILL on stop.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$", max_length=128)
    argv: list[str] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None
    uid: Optional[int] = Field(default=None, gt=0, lt=2**32 - 1, strict=True)
    gid: Optional[int] = Field(default=None, gt=0, lt=2**32 - 1, strict=True)
    auto_uid: bool = True
    network_isolated: bool = False
    restart_policy: RestartPolicy = RestartPolicy.NEVER
    max_retries: int = Field(default=3, ge=0)
    backoff_s: float = Field(default=0.5, gt=0)
    stop_grace_s: float = Field(default=5.0, ge=0)

    @model_validator(mode="after")
    def validate_process_settings(self) -> TaskSpec:
        if (self.uid is None) != (self.gid is None):
            raise ValueError("uid and gid must be supplied together")
        if not self.argv[0] or any("\0" in arg for arg in self.argv):
            raise ValueError("argv requires a nonempty executable and no NUL bytes")
        if self.cwd is not None and (not self.cwd.startswith("/") or "\0" in self.cwd):
            raise ValueError("cwd must be an absolute path without NUL bytes")
        if any(
            not key or "=" in key or "\0" in key or "\0" in value
            for key, value in self.env.items()
        ):
            raise ValueError("invalid child environment")
        return self


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

    ``seq`` increases within one daemon lifetime. The event buffer is bounded
    and resets with the daemon; this is not a durable or gap-free stream.
    """

    seq: int = 0
    ts: float = Field(default_factory=time.time)
    task: str
    kind: str
    detail: Optional[str] = None
