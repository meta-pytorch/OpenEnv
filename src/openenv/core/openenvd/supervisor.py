# SPDX-License-Identifier: BSD-3-Clause

"""Async process supervisor for openenvd sidecar tasks.

Each task is spawned in its own session (``start_new_session=True``) so the
supervisor can terminate the whole process tree with a single ``killpg``.
Restart behavior follows the task's ``RestartPolicy``; explicit stops never
trigger a restart.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections import deque
from typing import Optional

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

logger = logging.getLogger(__name__)

_MAX_BACKOFF_S = 30.0


class _TaskEntry:
    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.state = TaskState.REGISTERED
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.exit_code: Optional[int] = None
        self.restarts = 0
        self.stop_requested = False
        self.monitor: Optional[asyncio.Task] = None


class Supervisor:
    """Registers, spawns, supervises, and stops sidecar tasks."""

    def __init__(
        self,
        capabilities: Optional[IsolationCapabilities] = None,
        event_buffer_size: int = 1000,
    ):
        self._caps = capabilities or detect_capabilities()
        self._tasks: dict[str, _TaskEntry] = {}
        self._events: deque[SupervisorEvent] = deque(maxlen=event_buffer_size)

    async def register(self, spec: TaskSpec, autostart: bool = True) -> TaskStatus:
        if spec.name in self._tasks:
            raise ValueError(f"task {spec.name!r} is already registered")
        entry = _TaskEntry(spec)
        self._tasks[spec.name] = entry
        self._record(spec.name, "registered")
        if autostart:
            return await self.start(spec.name)
        return self.status(spec.name)

    async def start(self, name: str) -> TaskStatus:
        entry = self._get(name)
        if entry.state == TaskState.RUNNING:
            return self.status(name)

        entry.stop_requested = False
        entry.exit_code = None
        entry.state = TaskState.STARTING

        preexec, warnings = build_preexec(entry.spec, self._caps)
        for warning in warnings:
            logger.warning("openenvd task %s: %s", name, warning)

        env = None
        if entry.spec.env:
            env = {**os.environ, **entry.spec.env}
        try:
            proc = await asyncio.create_subprocess_exec(
                *entry.spec.argv,
                env=env,
                cwd=entry.spec.cwd,
                preexec_fn=preexec,
                start_new_session=True,
            )
        except OSError as e:
            entry.state = TaskState.FAILED
            self._record(name, "spawn_failed", str(e))
            raise

        entry.proc = proc
        entry.state = TaskState.RUNNING
        self._record(name, "started", f"pid={proc.pid}")
        entry.monitor = asyncio.create_task(self._monitor(entry))
        return self.status(name)

    async def stop(self, name: str) -> TaskStatus:
        entry = self._get(name)
        if entry.state in (TaskState.REGISTERED,):
            entry.state = TaskState.STOPPED
            return self.status(name)
        if entry.state in (TaskState.STOPPED, TaskState.EXITED, TaskState.FAILED):
            return self.status(name)

        entry.stop_requested = True
        proc = entry.proc
        if proc is not None and proc.returncode is None:
            await self._terminate_group(proc, entry.spec.stop_grace_s)

        if entry.monitor is not None:
            await entry.monitor
        return self.status(name)

    async def unregister(self, name: str) -> None:
        entry = self._get(name)
        if entry.state in (TaskState.RUNNING, TaskState.STARTING, TaskState.RESTARTING):
            await self.stop(name)
        del self._tasks[name]

    def status(self, name: str) -> TaskStatus:
        entry = self._get(name)
        return TaskStatus(
            name=name,
            state=entry.state,
            pid=entry.proc.pid if entry.proc is not None else None,
            exit_code=entry.exit_code,
            restarts=entry.restarts,
        )

    def status_all(self) -> list[TaskStatus]:
        return [self.status(name) for name in self._tasks]

    def events(self) -> list[SupervisorEvent]:
        return list(self._events)

    async def shutdown(self) -> None:
        for name in list(self._tasks):
            entry = self._tasks[name]
            if entry.state in (
                TaskState.RUNNING,
                TaskState.STARTING,
                TaskState.RESTARTING,
            ):
                try:
                    await self.stop(name)
                except Exception:
                    logger.exception("error stopping task %s during shutdown", name)

    async def _monitor(self, entry: _TaskEntry) -> None:
        name = entry.spec.name
        proc = entry.proc
        assert proc is not None
        code = await proc.wait()
        entry.exit_code = code
        entry.proc = None

        if entry.stop_requested:
            entry.state = TaskState.STOPPED
            self._record(name, "stopped", f"exit_code={code}")
            return

        self._record(name, "exited", f"exit_code={code}")

        policy = entry.spec.restart_policy
        wants_restart = policy == RestartPolicy.ALWAYS or (
            policy == RestartPolicy.ON_FAILURE and code != 0
        )
        if not wants_restart:
            entry.state = TaskState.EXITED
            return

        if (
            policy == RestartPolicy.ON_FAILURE
            and entry.restarts >= entry.spec.max_retries
        ):
            entry.state = TaskState.FAILED
            self._record(name, "failed", f"retries exhausted ({code})")
            return

        delay = min(entry.spec.backoff_s * (2**entry.restarts), _MAX_BACKOFF_S)
        entry.restarts += 1
        entry.state = TaskState.RESTARTING
        self._record(name, "restart_scheduled", f"delay={delay:.3f}s")
        await asyncio.sleep(delay)
        if entry.stop_requested or name not in self._tasks:
            entry.state = TaskState.STOPPED
            return
        await self.start(name)

    async def _terminate_group(
        self, proc: asyncio.subprocess.Process, grace_s: float
    ) -> None:
        self._signal_group(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=max(grace_s, 0.0))
        except asyncio.TimeoutError:
            self._signal_group(proc.pid, signal.SIGKILL)
            await proc.wait()

    @staticmethod
    def _signal_group(pgid: int, sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warning("insufficient permission to signal process group %d", pgid)

    def _get(self, name: str) -> _TaskEntry:
        try:
            return self._tasks[name]
        except KeyError:
            raise KeyError(f"unknown task: {name}") from None

    def _record(self, task: str, kind: str, detail: Optional[str] = None) -> None:
        self._events.append(SupervisorEvent(task=task, kind=kind, detail=detail))
