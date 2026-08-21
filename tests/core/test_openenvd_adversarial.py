# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial isolation tests for openenvd.

These tests attack the sidecar boundary from the workload's perspective:
reach the control surface through the network namespace, signal the daemon
or sibling tasks, and read privileged assets. Root-gated tests exercise the
real mechanisms (netns, UID drop); they are skipped when unprivileged.
"""

import asyncio
import json
import os
import socket

import httpx
import pytest
from openenv.core.openenvd import isolation
from openenv.core.openenvd.daemon import create_app
from openenv.core.openenvd.models import TaskSpec
from openenv.core.openenvd.supervisor import Supervisor

NOBODY_UID = 65534
NOGROUP_GID = 65534

requires_root = pytest.mark.skipif(os.geteuid() != 0, reason="requires root")


def _in_initial_user_ns() -> bool:
    try:
        with open("/proc/self/uid_map") as f:
            first = f.readline().split()
    except OSError:
        return False
    return first == ["0", "0", "4294967295"]


requires_real_root = pytest.mark.skipif(
    os.geteuid() != 0 or not _in_initial_user_ns(),
    reason="requires real root (initial user ns) for setuid to arbitrary uids",
)


async def _run_probe(spec: TaskSpec, timeout: float = 15.0):
    preexec, warnings = isolation.build_preexec(spec)
    assert warnings == [], f"probe setup lost isolation: {warnings}"
    proc = await asyncio.create_subprocess_exec(
        *spec.argv,
        preexec_fn=preexec,
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise AssertionError(f"probe {spec.argv} timed out")
    return proc.returncode, out.decode()


def _connect_probe(host: str, port: int) -> list[str]:
    script = (
        "import socket,sys;"
        "s=socket.socket();"
        "s.settimeout(2);"
        f"s.connect(({host!r},{port}));"
        "print('CONNECTED')"
    )
    return ["python3", "-c", script]


class TestControlSurfaceAuth:
    async def test_health_is_open(self):
        app = create_app(auth_token="secret")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://openenvd"
        ) as c:
            assert (await c.get("/health")).status_code == 200
        await app.supervisor.shutdown()

    @pytest.mark.parametrize("headers", [{}, {"authorization": "Bearer wrong"}])
    async def test_mutating_endpoints_reject_bad_token(self, headers):
        app = create_app(supervisor=Supervisor(), auth_token="secret")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://openenvd"
        ) as c:
            assert (await c.get("/tasks", headers=headers)).status_code == 401
            assert (await c.get("/events", headers=headers)).status_code == 401
            resp = await c.post(
                "/tasks",
                json={"name": "evil", "argv": ["sleep", "30"]},
                headers=headers,
            )
            assert resp.status_code == 401
            assert (
                await c.get("/tasks", headers={"authorization": "Bearer secret"})
            ).json() == []
        await app.supervisor.shutdown()

    async def test_valid_token_grants_access(self):
        app = create_app(supervisor=Supervisor(), auth_token="secret")
        transport = httpx.ASGITransport(app=app)
        headers = {"authorization": "Bearer secret"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://openenvd"
        ) as c:
            assert (
                await c.post(
                    "/tasks",
                    json={"name": "ok", "argv": ["true"], "autostart": False},
                    headers=headers,
                )
            ).status_code == 201
            assert (await c.get("/tasks", headers=headers)).status_code == 200
        await app.supervisor.shutdown()

    async def test_no_token_configured_keeps_api_open(self):
        app = create_app(supervisor=Supervisor())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://openenvd"
        ) as c:
            assert (await c.get("/tasks")).status_code == 200
        await app.supervisor.shutdown()


@requires_root
class TestNetworkIsolationBlocksControlSurface:
    async def test_isolated_child_cannot_reach_daemon_listener(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            spec = TaskSpec(
                name="probe",
                argv=_connect_probe("127.0.0.1", port),
                network_isolated=True,
            )
            rc, out = await _run_probe(spec)
            assert rc != 0, f"child reached daemon listener: {out}"
            assert "CONNECTED" not in out
        finally:
            listener.close()

    async def test_unisolated_child_can_reach_listener_control(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            spec = TaskSpec(name="probe", argv=_connect_probe("127.0.0.1", port))
            rc, out = await _run_probe(spec)
            assert rc == 0 and "CONNECTED" in out, (
                "control failed: shared-netns child should reach the listener"
            )
        finally:
            listener.close()

    async def test_isolated_child_sees_only_loopback(self):
        script = (
            "import socket,json;print(json.dumps([n for _,n in socket.if_nameindex()]))"
        )
        spec = TaskSpec(
            name="probe",
            argv=["python3", "-c", script],
            network_isolated=True,
        )
        rc, out = await _run_probe(spec)
        assert rc == 0
        interfaces = json.loads(out.strip().splitlines()[-1])
        assert interfaces == ["lo"], f"unexpected interfaces visible: {interfaces}"


@requires_real_root
class TestUidSeparationBlocksPrivilege:
    async def test_workload_cannot_read_daemon_owned_assets(self):
        path = "/tmp/openenvd-adversarial-asset"
        with open(path, "w") as f:
            f.write("SECRET_SOLUTION")
        os.chmod(path, 0o700)
        try:
            spec = TaskSpec(
                name="probe",
                argv=["cat", path],
                uid=NOBODY_UID,
                gid=NOGROUP_GID,
            )
            rc, out = await _run_probe(spec)
            assert rc != 0, "workload read privileged asset"
            assert "SECRET" not in out
        finally:
            os.unlink(path)

    async def test_workload_cannot_signal_the_daemon(self):
        target_pid = os.getpid()
        script = (
            "import os,sys;"
            "try:"
            f"    os.kill({target_pid}, 0);"
            '    print("SIGNALABLE")'
            "except PermissionError:"
            '    print("DENIED")'
        )
        spec = TaskSpec(
            name="probe",
            argv=["python3", "-c", script],
            uid=NOBODY_UID,
            gid=NOGROUP_GID,
        )
        rc, out = await _run_probe(spec)
        assert rc == 0
        assert "DENIED" in out, "workload can signal the daemon"

    async def test_sibling_tasks_with_distinct_uids_cannot_signal_each_other(self):
        script = "import os,sys,time;time.sleep(30)"
        supervisor = Supervisor()
        victim = await supervisor.register(
            TaskSpec(
                name="victim",
                argv=["sh", "-c", script],
                uid=NOBODY_UID,
                gid=NOGROUP_GID,
            )
        )
        for _ in range(100):
            if victim.pid is not None:
                break
            await asyncio.sleep(0.05)
        attacker_spec = TaskSpec(
            name="attacker",
            argv=[
                "python3",
                "-c",
                "import os;"
                "try:"
                f"    os.kill({victim.pid}, 0);"
                '    print("SIGNALABLE")'
                "except PermissionError:"
                '    print("DENIED")',
            ],
            uid=65533,
            gid=65533,
        )
        try:
            rc, out = await _run_probe(attacker_spec)
            assert rc == 0
            assert "DENIED" in out, "task signaled a sibling task"
        finally:
            await supervisor.shutdown()


@requires_root
class TestSupervisorSurvivesWorkloadHostility:
    async def test_crash_loop_does_not_break_supervision(self):
        supervisor = Supervisor()
        spec = TaskSpec(
            name="hostile",
            argv=["sh", "-c", "exit 1"],
            restart_policy="on_failure",
            max_retries=5,
            backoff_s=0.05,
        )
        await supervisor.register(spec)
        for _ in range(100):
            if supervisor.status("hostile").restarts >= 2:
                break
            await asyncio.sleep(0.05)
        assert supervisor.status_all(), "supervisor lost its registry"
        await supervisor.shutdown()
        assert supervisor.status("hostile").state != "running"

    async def test_daemon_serves_health_during_crash_loop(self):
        app = create_app()
        supervisor = app.supervisor
        await supervisor.register(
            TaskSpec(
                name="loop",
                argv=["false"],
                restart_policy="always",
                backoff_s=0.02,
            )
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://openenvd"
        ) as c:
            for _ in range(10):
                resp = await c.get("/health")
                assert resp.status_code == 200
                await asyncio.sleep(0.03)
        await supervisor.shutdown()
