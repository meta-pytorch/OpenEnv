# SPDX-License-Identifier: BSD-3-Clause

"""One lifecycle loop per registered process, including restart and cleanup.

Tasks run in separate process groups. Group cleanup covers ordinary descendants;
the enclosing container must contain processes that deliberately leave the group.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections import deque
from typing import Optional

from openenv.core.openenvd.isolation import (
    detect_capabilities,
    IsolationCapabilities,
    IsolationError,
    spawn_task,
    validate_isolation,
)
from openenv.core.openenvd.models import (
    RestartPolicy,
    SupervisorEvent,
    TaskSpec,
    TaskState,
    TaskStatus,
)
from openenv.core.openenvd.uid_allocator import UidAllocator

_MAX_BACKOFF_S = 30.0


class _TaskEntry:
    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.state = TaskState.REGISTERED
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.exit_code: Optional[int] = None
        self.restarts = 0
        self.lock = asyncio.Lock()
        self.stop = asyncio.Event()
        self.ready = asyncio.Event()
        self.runner: Optional[asyncio.Task] = None
        self.spawn_error: Optional[Exception] = None


class Supervisor:
    """Supervise operator-supplied tasks inside an existing container boundary."""

    def __init__(
        self,
        capabilities: Optional[IsolationCapabilities] = None,
        event_buffer_size: int = 1000,
        uid_allocator: Optional[UidAllocator] = None,
    ):
        if event_buffer_size <= 0:
            raise ValueError("event_buffer_size must be positive")
        self._caps = capabilities or detect_capabilities()
        self._tasks: dict[str, _TaskEntry] = {}
        self._events: deque[SupervisorEvent] = deque(maxlen=event_buffer_size)
        self._next_seq = 0
        self._uids = uid_allocator or UidAllocator()
        self._closed = False

    async def register(self, spec: TaskSpec, autostart: bool = True) -> TaskStatus:
        self._check_open()
        spec = TaskSpec.model_validate(spec.model_dump())
        if spec.name in self._tasks:
            raise ValueError("task is already registered")
        if spec.auto_uid and spec.uid is None:
            if not self._caps.can_allocate_uids:
                raise IsolationError("dedicated task UIDs are unavailable")
            try:
                spec.uid, spec.gid = self._uids.acquire(spec.name)
            except ValueError as exc:
                raise IsolationError("dedicated task UID range exhausted") from exc
        elif spec.uid is not None:
            try:
                self._uids.reserve(spec.name, spec.uid)
            except ValueError as exc:
                raise IsolationError("task UID is already assigned") from exc
        try:
            validate_isolation(spec, self._caps)
        except Exception:
            self._uids.release(spec.name)
            raise
        self._tasks[spec.name] = _TaskEntry(spec)
        self._record(spec.name, "registered")
        if autostart:
            return await self.start(spec.name)
        return self.status(spec.name)

    async def start(self, name: str) -> TaskStatus:
        entry = self._get(name)
        async with entry.lock:
            self._check_open()
            self._check_entry(name, entry)
            if entry.runner is not None and not entry.runner.done():
                return self.status(name)
            if entry.proc is not None:
                raise IsolationError(
                    "previous task cleanup failed; stop it before starting"
                )
            entry.stop.clear()
            entry.ready.clear()
            entry.spawn_error = None
            entry.exit_code = None
            entry.restarts = 0
            entry.state = TaskState.STARTING
            # The runner owns the child even if the requesting HTTP client leaves.
            entry.runner = asyncio.create_task(self._run(entry))
            await entry.ready.wait()
            if entry.spawn_error is not None:
                raise entry.spawn_error
            return self.status(name)

    async def stop(self, name: str) -> TaskStatus:
        entry = self._get(name)
        async with entry.lock:
            self._check_entry(name, entry)
            await self._stop(entry)
            return self.status(name)

    async def unregister(self, name: str) -> None:
        entry = self._get(name)
        async with entry.lock:
            self._check_entry(name, entry)
            await self._stop(entry)
            self._uids.release(name)
            del self._tasks[name]

    async def _stop(self, entry: _TaskEntry) -> None:
        entry.stop.set()
        if entry.runner is not None:
            # A cancelled runner must not cancel the stop request.
            # A child retained after failed cleanup must still be retried below.
            await asyncio.shield(asyncio.gather(entry.runner, return_exceptions=True))
        retained_child = entry.proc is not None
        await self._clean_up(entry)
        if retained_child or entry.state == TaskState.REGISTERED:
            entry.state = TaskState.STOPPED
            self._record(entry.spec.name, "stopped")

    def status(self, name: str) -> TaskStatus:
        entry = self._get(name)
        return TaskStatus(
            name=name,
            state=entry.state,
            pid=entry.proc.pid if entry.proc is not None else None,
            exit_code=entry.exit_code,
            restarts=entry.restarts,
            uid=entry.spec.uid,
            gid=entry.spec.gid,
        )

    def status_all(self) -> list[TaskStatus]:
        return [self.status(name) for name in self._tasks]

    def events(self, after: int = -1) -> list[SupervisorEvent]:
        return [event.model_copy() for event in self._events if event.seq > after]

    async def shutdown(self) -> None:
        self._closed = True
        # Wake every task immediately, including those waiting to restart.
        entries = list(self._tasks.values())
        for entry in entries:
            entry.stop.set()
        await asyncio.gather(*(self._stop(entry) for entry in entries))

    async def _run(self, entry: _TaskEntry) -> None:
        spec = entry.spec
        delay = min(spec.backoff_s, _MAX_BACKOFF_S)
        try:
            while not entry.stop.is_set():
                entry.proc = await spawn_task(
                    spec, self._caps, env={"PATH": os.defpath, **spec.env}
                )
                entry.state = TaskState.RUNNING
                self._record(spec.name, "started", f"pid={entry.proc.pid}")
                entry.ready.set()
                await self._wait_and_clean_up(entry)
                if entry.stop.is_set():
                    break
                self._record(spec.name, "exited", f"exit_code={entry.exit_code}")
                if spec.restart_policy == RestartPolicy.NEVER or (
                    spec.restart_policy == RestartPolicy.ON_FAILURE
                    and entry.exit_code == 0
                ):
                    entry.state = TaskState.EXITED
                    return
                if (
                    spec.restart_policy == RestartPolicy.ON_FAILURE
                    and entry.restarts >= spec.max_retries
                ):
                    entry.state = TaskState.FAILED
                    self._record(spec.name, "failed", "restart budget exhausted")
                    return
                entry.state = TaskState.RESTARTING
                self._record(spec.name, "restart_scheduled", f"delay={delay:.3f}s")
                try:
                    await asyncio.wait_for(entry.stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    entry.restarts += 1
                    delay = min(delay * 2, _MAX_BACKOFF_S)
            entry.state = TaskState.STOPPED
            self._record(spec.name, "stopped", f"exit_code={entry.exit_code}")
        except asyncio.CancelledError:
            entry.stop.set()
            raise
        except Exception as exc:
            entry.spawn_error = exc
            entry.state = TaskState.FAILED
            # Exception messages can include arguments, paths, or secrets.
            self._record(spec.name, "failed", type(exc).__name__)
        finally:
            try:
                await self._clean_up(entry)
                if entry.stop.is_set() and entry.state != TaskState.STOPPED:
                    entry.state = TaskState.STOPPED
                    self._record(spec.name, "stopped", f"exit_code={entry.exit_code}")
            finally:
                entry.ready.set()

    async def _wait_and_clean_up(self, entry: _TaskEntry) -> None:
        proc = entry.proc
        assert proc is not None
        exited = asyncio.create_task(proc.wait())
        stopped = asyncio.create_task(entry.stop.wait())
        try:
            await asyncio.wait((exited, stopped), return_when=asyncio.FIRST_COMPLETED)
            # The group may outlive its leader, even after a successful exit.
            await self._clean_up(entry)
        finally:
            exited.cancel()
            stopped.cancel()
            await asyncio.gather(exited, stopped, return_exceptions=True)

    async def _clean_up(self, entry: _TaskEntry) -> None:
        proc = entry.proc
        if proc is not None:
            await self._terminate_group(proc, entry.spec.stop_grace_s)
            entry.exit_code = proc.returncode
            entry.proc = None

    async def _terminate_group(
        self, proc: asyncio.subprocess.Process, grace_s: float
    ) -> None:
        if self._signal_group(proc.pid, signal.SIGTERM):
            deadline = asyncio.get_running_loop().time() + grace_s
            while self._signal_group(proc.pid, 0):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    self._signal_group(proc.pid, signal.SIGKILL)
                    break
                await asyncio.sleep(min(remaining, 0.02))
        await proc.wait()

    @staticmethod
    def _signal_group(pgid: int, sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False

    def _get(self, name: str) -> _TaskEntry:
        try:
            return self._tasks[name]
        except KeyError:
            raise KeyError("unknown task") from None

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("supervisor is shut down")

    def _check_entry(self, name: str, entry: _TaskEntry) -> None:
        if self._get(name) is not entry:
            raise KeyError("task was replaced")

    def _record(self, task: str, kind: str, detail: Optional[str] = None) -> None:
        self._events.append(
            SupervisorEvent(seq=self._next_seq, task=task, kind=kind, detail=detail)
        )
        self._next_seq += 1
