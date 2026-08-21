# SPDX-License-Identifier: BSD-3-Clause

"""Tests for dedicated-UID allocation of supervised tasks."""

import asyncio

from openenv.core.openenvd.isolation import IsolationCapabilities
from openenv.core.openenvd.models import TaskSpec, TaskState
from openenv.core.openenvd.supervisor import Supervisor
from openenv.core.openenvd.uid_allocator import UidAllocator

ROOT_CAPS = IsolationCapabilities(
    can_drop_uid=True, can_unshare_net=False, can_allocate_uids=True
)
USER_CAPS = IsolationCapabilities(can_drop_uid=False, can_unshare_net=False)


class TestUidAllocator:
    def test_allocates_lowest_free_pair(self):
        alloc = UidAllocator(base=1000, count=8)
        assert alloc.acquire("a") == (1000, 1000)
        assert alloc.acquire("b") == (1001, 1001)

    def test_acquire_is_stable_per_task(self):
        alloc = UidAllocator(base=1000, count=8)
        assert alloc.acquire("a") == (1000, 1000)
        assert alloc.acquire("a") == (1000, 1000)

    def test_release_returns_uid_to_pool(self):
        alloc = UidAllocator(base=1000, count=2)
        alloc.acquire("a")
        alloc.acquire("b")
        assert alloc.available == 0
        assert alloc.acquire("c") is None
        alloc.release("a")
        assert alloc.acquire("c") == (1000, 1000)

    def test_release_unknown_task_is_noop(self):
        alloc = UidAllocator()
        alloc.release("ghost")
        assert alloc.available == alloc._count


class TestSupervisorAutoUid:
    async def test_privileged_daemon_assigns_distinct_uids(self):
        supervisor = Supervisor(capabilities=ROOT_CAPS)
        a = await supervisor.register(
            TaskSpec(name="a", argv=["sleep", "30"]), autostart=False
        )
        b = await supervisor.register(
            TaskSpec(name="b", argv=["sleep", "30"]), autostart=False
        )
        assert a.uid is not None
        assert b.uid is not None
        assert a.uid != b.uid
        await supervisor.shutdown()

    async def test_unprivileged_daemon_skips_allocation(self):
        supervisor = Supervisor(capabilities=USER_CAPS)
        status = await supervisor.register(
            TaskSpec(name="a", argv=["true"], autostart=False), autostart=False
        )
        assert status.uid is None
        await supervisor.shutdown()

    async def test_explicit_uid_wins(self):
        supervisor = Supervisor(capabilities=ROOT_CAPS)
        status = await supervisor.register(
            TaskSpec(name="a", argv=["true"], uid=1234, gid=1234),
            autostart=False,
        )
        assert status.uid == 1234
        await supervisor.shutdown()

    async def test_auto_uid_opt_out(self):
        supervisor = Supervisor(capabilities=ROOT_CAPS)
        status = await supervisor.register(
            TaskSpec(name="a", argv=["true"], auto_uid=False), autostart=False
        )
        assert status.uid is None
        await supervisor.shutdown()

    async def test_unregister_releases_uid_for_reuse(self):
        supervisor = Supervisor(capabilities=ROOT_CAPS)
        first = await supervisor.register(
            TaskSpec(name="a", argv=["true"], autostart=False), autostart=False
        )
        await supervisor.unregister("a")
        second = await supervisor.register(
            TaskSpec(name="b", argv=["true"], autostart=False), autostart=False
        )
        assert second.uid == first.uid
        await supervisor.shutdown()

    async def test_uid_survives_restart_of_same_task(self, monkeypatch):
        import openenv.core.openenvd.supervisor as sup_mod

        monkeypatch.setattr(sup_mod, "build_preexec", lambda spec, caps: (None, []))
        supervisor = Supervisor(capabilities=ROOT_CAPS)
        spec = TaskSpec(
            name="crasher",
            argv=["sh", "-c", "exit 1"],
            restart_policy="on_failure",
            max_retries=1,
            backoff_s=0.05,
        )
        before = await supervisor.register(spec)
        for _ in range(100):
            if supervisor.status("crasher").state == TaskState.FAILED:
                break
            await asyncio.sleep(0.05)
        after = supervisor.status("crasher")
        assert before.uid is not None
        assert after.uid == before.uid
        await supervisor.shutdown()
