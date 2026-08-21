# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenvd process supervisor."""

import os
import subprocess

import pytest
from openenv.core.openenvd.models import RestartPolicy, TaskSpec, TaskState
from openenv.core.openenvd.supervisor import Supervisor


async def wait_for_state(supervisor, name, state, timeout=10.0):
    async for _ in _poll(timeout):
        status = supervisor.status(name)
        if status.state == state:
            return status
    raise AssertionError(f"task {name} never reached {state}")


async def _poll(timeout):
    import asyncio

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        yield
        await asyncio.sleep(0.05)


class TestSupervisorBasics:
    async def test_start_long_running_task_reports_running(self):
        supervisor = Supervisor()
        spec = TaskSpec(name="sleeper", argv=["sleep", "30"])
        status = await supervisor.register(spec, autostart=True)
        assert status.state in (TaskState.STARTING, TaskState.RUNNING)

        status = await wait_for_state(supervisor, "sleeper", TaskState.RUNNING)
        assert status.pid is not None
        os.kill(status.pid, 0)

        await supervisor.shutdown()

    async def test_exit_code_captured(self):
        supervisor = Supervisor()
        spec = TaskSpec(name="exiter", argv=["sh", "-c", "exit 7"])
        await supervisor.register(spec, autostart=True)
        status = await wait_for_state(supervisor, "exiter", TaskState.EXITED)
        assert status.exit_code == 7
        assert status.pid is None or not _pid_alive(status.pid)
        await supervisor.shutdown()

    async def test_duplicate_name_rejected(self):
        supervisor = Supervisor()
        await supervisor.register(TaskSpec(name="dup", argv=["sleep", "30"]))
        with pytest.raises(ValueError):
            await supervisor.register(
                TaskSpec(name="dup", argv=["sleep", "30"]), autostart=False
            )
        await supervisor.shutdown()

    async def test_unknown_task_lookup_raises(self):
        supervisor = Supervisor()
        with pytest.raises(KeyError):
            supervisor.status("nope")

    async def test_status_all_lists_tasks(self):
        supervisor = Supervisor()
        await supervisor.register(
            TaskSpec(name="a", argv=["sleep", "30"]), autostart=False
        )
        await supervisor.register(
            TaskSpec(name="b", argv=["sleep", "30"]), autostart=False
        )
        names = {s.name for s in supervisor.status_all()}
        assert names == {"a", "b"}
        states = {s.state for s in supervisor.status_all()}
        assert states == {TaskState.REGISTERED}
        await supervisor.shutdown()


class TestRestartPolicies:
    async def test_never_policy_leaves_exited_task_dead(self):
        supervisor = Supervisor()
        spec = TaskSpec(
            name="once",
            argv=["sh", "-c", "exit 0"],
            restart_policy=RestartPolicy.NEVER,
        )
        await supervisor.register(spec, autostart=True)
        status = await wait_for_state(supervisor, "once", TaskState.EXITED)
        assert status.restarts == 0
        await supervisor.shutdown()

    async def test_on_failure_restarts_then_fails_after_max_retries(self):
        supervisor = Supervisor()
        spec = TaskSpec(
            name="crasher",
            argv=["sh", "-c", "exit 1"],
            restart_policy=RestartPolicy.ON_FAILURE,
            max_retries=2,
            backoff_s=0.05,
        )
        await supervisor.register(spec, autostart=True)
        status = await wait_for_state(supervisor, "crasher", TaskState.FAILED)
        assert status.restarts == 2
        assert status.exit_code == 1
        await supervisor.shutdown()

    async def test_on_failure_does_not_restart_clean_exit(self):
        supervisor = Supervisor()
        spec = TaskSpec(
            name="clean",
            argv=["true"],
            restart_policy=RestartPolicy.ON_FAILURE,
            max_retries=3,
            backoff_s=0.05,
        )
        await supervisor.register(spec, autostart=True)
        status = await wait_for_state(supervisor, "clean", TaskState.EXITED)
        assert status.restarts == 0
        await supervisor.shutdown()

    async def test_always_policy_restarts_clean_exits(self):
        supervisor = Supervisor()
        spec = TaskSpec(
            name="phoenix",
            argv=["true"],
            restart_policy=RestartPolicy.ALWAYS,
            backoff_s=0.05,
        )
        await supervisor.register(spec, autostart=True)
        async for _ in _poll(10.0):
            if supervisor.status("phoenix").restarts >= 2:
                break
        assert supervisor.status("phoenix").restarts >= 2
        await supervisor.shutdown()
        assert supervisor.status("phoenix").state != TaskState.RUNNING


class TestStop:
    async def test_stop_terminates_running_task(self):
        supervisor = Supervisor()
        await supervisor.register(TaskSpec(name="sleeper", argv=["sleep", "30"]))
        await wait_for_state(supervisor, "sleeper", TaskState.RUNNING)
        status = await supervisor.stop("sleeper")
        assert status.state == TaskState.STOPPED
        await supervisor.shutdown()

    async def test_stop_kills_entire_process_group(self):
        supervisor = Supervisor()
        spec = TaskSpec(
            name="tree",
            argv=["sh", "-c", "sleep 300 & wait"],
            stop_grace_s=1.0,
        )
        await supervisor.register(spec)
        await wait_for_state(supervisor, "tree", TaskState.RUNNING)

        pgrep = subprocess.run(
            ["pgrep", "-f", "sleep 300"], capture_output=True, text=True
        )
        grandchild_pids = pgrep.stdout.split()
        assert grandchild_pids, "grandchild sleep process not found"

        await supervisor.stop("tree")
        await supervisor.shutdown()

        for pid in grandchild_pids:
            assert not _pid_alive(int(pid)), f"grandchild {pid} survived"

    async def test_stop_escapes_term_ignored_child(self):
        supervisor = Supervisor()
        spec = TaskSpec(
            name="stubborn",
            argv=["sh", "-c", 'trap "" TERM; sleep 300'],
            stop_grace_s=0.5,
        )
        await supervisor.register(spec)
        await wait_for_state(supervisor, "stubborn", TaskState.RUNNING)
        status = await supervisor.stop("stubborn")
        assert status.state == TaskState.STOPPED
        await supervisor.shutdown()

    async def test_stop_does_not_trigger_restart(self):
        supervisor = Supervisor()
        spec = TaskSpec(
            name="restarted",
            argv=["sleep", "30"],
            restart_policy=RestartPolicy.ALWAYS,
            backoff_s=0.05,
        )
        await supervisor.register(spec)
        await wait_for_state(supervisor, "restarted", TaskState.RUNNING)
        status = await supervisor.stop("restarted")
        assert status.state == TaskState.STOPPED
        await _sleep(0.3)
        assert supervisor.status("restarted").restarts == 0
        await supervisor.shutdown()

    async def test_unregister_stops_and_removes(self):
        supervisor = Supervisor()
        await supervisor.register(TaskSpec(name="gone", argv=["sleep", "30"]))
        await wait_for_state(supervisor, "gone", TaskState.RUNNING)
        await supervisor.unregister("gone")
        with pytest.raises(KeyError):
            supervisor.status("gone")
        await supervisor.shutdown()


class TestEvents:
    async def test_events_recorded(self):
        supervisor = Supervisor()
        await supervisor.register(TaskSpec(name="ev", argv=["sh", "-c", "exit 0"]))
        await wait_for_state(supervisor, "ev", TaskState.EXITED)
        events = supervisor.events()
        kinds = [e.kind for e in events]
        assert "started" in kinds
        assert "exited" in kinds
        await supervisor.shutdown()


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


async def _sleep(seconds):
    import asyncio

    await asyncio.sleep(seconds)
