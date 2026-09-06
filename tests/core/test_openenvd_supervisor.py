# SPDX-License-Identifier: BSD-3-Clause

"""Exercise lifecycle ownership with real child processes."""

import asyncio
import json
import os
import subprocess
import sys

import pytest
from openenv.core.openenvd import supervisor as supervisor_module
from openenv.core.openenvd.models import TaskSpec, TaskState
from openenv.core.openenvd.supervisor import Supervisor


@pytest.fixture
async def supervisor():
    instance = Supervisor()
    try:
        yield instance
    finally:
        await instance.shutdown()


def task(name="task", argv=None, **kwargs):
    # These tests run trusted local commands without a separate OS identity.
    return TaskSpec(name=name, argv=argv or ["sleep", "30"], auto_uid=False, **kwargs)


async def wait_for_state(supervisor, name, state, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        status = supervisor.status(name)
        if status.state == state:
            return status
        await asyncio.sleep(0.01)
    raise AssertionError(f"task {name} never reached {state}")


async def test_start_and_stop(supervisor):
    running = await supervisor.register(task())
    assert running.state == TaskState.RUNNING
    os.kill(running.pid, 0)
    stopped = await supervisor.stop("task")
    assert stopped.state == TaskState.STOPPED
    assert stopped.pid is None


async def test_registered_task_can_be_stopped_without_starting(supervisor):
    await supervisor.register(task(), autostart=False)
    assert (await supervisor.stop("task")).state == TaskState.STOPPED
    assert not any(event.kind == "started" for event in supervisor.events())


async def test_duplicate_and_unknown_tasks(supervisor):
    await supervisor.register(task(), autostart=False)
    with pytest.raises(ValueError):
        await supervisor.register(task())
    with pytest.raises(KeyError):
        await supervisor.start("missing")
    await supervisor.unregister("task")
    assert supervisor.status_all() == []


@pytest.mark.parametrize(
    "policy,code,retries,state",
    [
        ("never", 7, 0, TaskState.EXITED),
        ("on_failure", 0, 0, TaskState.EXITED),
        ("on_failure", 7, 2, TaskState.FAILED),
    ],
)
async def test_restart_policy(supervisor, policy, code, retries, state):
    await supervisor.register(
        task(
            argv=["sh", "-c", f"exit {code}"],
            restart_policy=policy,
            max_retries=2,
            backoff_s=0.01,
        )
    )
    status = await wait_for_state(supervisor, "task", state)
    assert status.exit_code == code
    assert status.restarts == retries
    assert status.pid is None


async def test_manual_start_renews_restart_budget(supervisor):
    await supervisor.register(
        task(argv=["false"], restart_policy="on_failure", max_retries=1, backoff_s=0.01)
    )
    await wait_for_state(supervisor, "task", TaskState.FAILED)
    await supervisor.start("task")
    status = await wait_for_state(supervisor, "task", TaskState.FAILED)
    assert status.restarts == 1
    assert sum(event.kind == "started" for event in supervisor.events()) == 4


async def test_stop_interrupts_long_restart_backoff(supervisor):
    await supervisor.register(
        task(argv=["true"], restart_policy="always", backoff_s=30)
    )
    await wait_for_state(supervisor, "task", TaskState.RESTARTING)
    status = await asyncio.wait_for(supervisor.stop("task"), timeout=0.5)
    assert status.state == TaskState.STOPPED
    assert status.restarts == 0


async def test_concurrent_starts_create_one_process(supervisor, monkeypatch):
    original = supervisor_module.spawn_task
    calls = 0

    async def slow_spawn(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return await original(*args, **kwargs)

    monkeypatch.setattr(supervisor_module, "spawn_task", slow_spawn)
    await supervisor.register(task(), autostart=False)
    statuses = await asyncio.gather(*(supervisor.start("task") for _ in range(5)))
    assert calls == 1
    assert len({status.pid for status in statuses}) == 1


async def test_start_during_backoff_does_not_create_another_runner(supervisor):
    await supervisor.register(
        task(argv=["true"], restart_policy="always", backoff_s=30)
    )
    await wait_for_state(supervisor, "task", TaskState.RESTARTING)
    assert (await supervisor.start("task")).state == TaskState.RESTARTING
    assert sum(event.kind == "started" for event in supervisor.events()) == 1


async def test_cancelled_start_request_keeps_child_owned(supervisor, monkeypatch):
    original = supervisor_module.spawn_task
    spawning = asyncio.Event()
    proceed = asyncio.Event()

    async def delayed_spawn(*args, **kwargs):
        spawning.set()
        await proceed.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(supervisor_module, "spawn_task", delayed_spawn)
    request = asyncio.create_task(supervisor.register(task()))
    await spawning.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    proceed.set()
    await wait_for_state(supervisor, "task", TaskState.RUNNING)
    assert (await supervisor.stop("task")).state == TaskState.STOPPED


@pytest.mark.parametrize("operation", ["start", "stop", "unregister"])
async def test_queued_operation_cannot_touch_replacement_task(supervisor, operation):
    await supervisor.register(task(), autostart=False)
    old_entry = supervisor._tasks["task"]
    await old_entry.lock.acquire()
    request = asyncio.create_task(getattr(supervisor, operation)("task"))
    try:
        await asyncio.sleep(0)
        assert not request.done()
        # Hold a queued request at the boundary between deleting the old task
        # and releasing its lock, then register another task with the same name.
        del supervisor._tasks["task"]
        replacement = await supervisor.register(task())
    finally:
        old_entry.lock.release()
    with pytest.raises(KeyError):
        await request
    assert supervisor.status("task") == replacement
    os.kill(replacement.pid, 0)


async def test_cancelled_runner_reaps_child_and_waiters(supervisor):
    existing_tasks = asyncio.all_tasks()
    running = await supervisor.register(task())
    runner = supervisor._tasks["task"].runner
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    status = supervisor.status("task")
    assert status.state == TaskState.STOPPED
    assert status.pid is None
    with pytest.raises(ProcessLookupError):
        os.kill(running.pid, 0)
    assert not (asyncio.all_tasks() - existing_tasks)
    # A completed cancellation must not make the registry impossible to manage.
    await supervisor.unregister("task")
    assert supervisor.status_all() == []


async def test_cleanup_failure_retains_child_and_can_be_retried(
    supervisor, monkeypatch
):
    running = await supervisor.register(task())
    process = supervisor._tasks["task"].proc

    def denied_signal(pgid, signal):
        raise PermissionError("cannot signal task")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(supervisor, "_signal_group", denied_signal)
            with pytest.raises(PermissionError):
                await supervisor.unregister("task")
            status = supervisor.status("task")
            assert status.state == TaskState.FAILED
            assert status.pid == running.pid
            os.kill(running.pid, 0)
            with pytest.raises(OSError):
                await supervisor.start("task")
            assert supervisor.status("task").pid == running.pid
        await supervisor.unregister("task")
        assert supervisor.status_all() == []
        with pytest.raises(ProcessLookupError):
            os.kill(running.pid, 0)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_shutdown_during_spawn_stops_child(supervisor, monkeypatch):
    original = supervisor_module.spawn_task
    spawning = asyncio.Event()
    proceed = asyncio.Event()

    async def delayed_spawn(*args, **kwargs):
        spawning.set()
        await proceed.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(supervisor_module, "spawn_task", delayed_spawn)
    request = asyncio.create_task(supervisor.register(task()))
    await spawning.wait()
    shutdown = asyncio.create_task(supervisor.shutdown())
    await asyncio.sleep(0)
    proceed.set()
    await asyncio.gather(request, shutdown)
    assert supervisor.status("task").state == TaskState.STOPPED
    with pytest.raises(ValueError):
        await supervisor.start("task")
    with pytest.raises(ValueError):
        await supervisor.register(task(name="new"))


async def test_spawn_failure_is_visible_and_does_not_leak_arguments(supervisor):
    with pytest.raises(OSError):
        await supervisor.register(task(argv=["/missing/secret-credential"]))
    assert supervisor.status("task").state == TaskState.FAILED
    assert "secret-credential" not in str(supervisor.events())


async def test_restart_spawn_failure_becomes_failed(supervisor, monkeypatch):
    original = supervisor_module.spawn_task
    calls = 0

    async def failing_restart(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("sensitive launch detail")
        return await original(*args, **kwargs)

    monkeypatch.setattr(supervisor_module, "spawn_task", failing_restart)
    await supervisor.register(
        task(argv=["false"], restart_policy="always", backoff_s=0.01)
    )
    await wait_for_state(supervisor, "task", TaskState.FAILED)
    assert "sensitive" not in str(supervisor.events())
    await supervisor.unregister("task")


async def test_environment_is_explicit_and_snapshot_is_private(
    supervisor, monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENENVD_TOKEN", "daemon-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cloud-secret")
    output = tmp_path / "result.json"
    script = (
        "import json,os,pathlib;"
        f"pathlib.Path({str(output)!r}).write_text("
        "json.dumps({'env': dict(os.environ), 'cwd': os.getcwd()}))"
    )
    spec = task(argv=[sys.executable, "-c", script], env={"TASK_SETTING": "original"})
    await supervisor.register(spec, autostart=False)
    spec.env["TASK_SETTING"] = "changed"
    spec.argv[:] = ["false"]
    await supervisor.start("task")
    await wait_for_state(supervisor, "task", TaskState.EXITED)
    result = json.loads(output.read_text())
    assert result["env"]["TASK_SETTING"] == "original"
    assert result["env"]["PATH"] == os.defpath
    assert "OPENENVD_TOKEN" not in result["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in result["env"]
    assert result["cwd"] == "/"
    assert output.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("leader_exits", [False, True])
async def test_cleanup_kills_stubborn_descendant_after_leader_exit(
    supervisor, tmp_path, leader_exits
):
    ready = tmp_path / "child.pid"
    script = f"""
import os, pathlib, signal, time
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))
    while True:
        time.sleep(1)
while not pathlib.Path({str(ready)!r}).exists():
    time.sleep(0.01)
if {leader_exits!r}:
    raise SystemExit(0)
while True:
    time.sleep(1)
"""
    await supervisor.register(
        task(argv=[sys.executable, "-c", script], stop_grace_s=0.1)
    )
    for _ in range(200):
        if ready.exists():
            break
        await asyncio.sleep(0.01)
    assert ready.exists(), "child did not signal readiness"
    pid = int(ready.read_text())
    if leader_exits:
        await wait_for_state(supervisor, "task", TaskState.EXITED)
    else:
        await supervisor.stop("task")
    # Linux may leave an orphaned zombie for container PID 1 to reap.
    for _ in range(100):
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        )
        if not result.stdout.strip() or result.stdout.strip().startswith("Z"):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail(f"descendant {pid} survived group cleanup")


async def test_events_are_bounded():
    supervisor = Supervisor(event_buffer_size=2)
    try:
        for name in ("first", "second", "third"):
            await supervisor.register(task(name=name), autostart=False)
        events = supervisor.events()
        assert [event.task for event in events] == ["second", "third"]
        assert [event.task for event in supervisor.events(after=events[0].seq)] == [
            "third"
        ]
    finally:
        await supervisor.shutdown()
