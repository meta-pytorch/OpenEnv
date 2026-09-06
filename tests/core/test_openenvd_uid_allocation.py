# SPDX-License-Identifier: BSD-3-Clause

"""Dedicated identities must not collide or be recycled while a daemon lives."""

import asyncio

import pytest
from openenv.core.openenvd.isolation import IsolationCapabilities, IsolationError
from openenv.core.openenvd.models import TaskSpec, TaskState
from openenv.core.openenvd.supervisor import Supervisor
from openenv.core.openenvd.uid_allocator import MAX_UID, UidAllocator


ROOT_CAPS = IsolationCapabilities(True, False, True)
USER_CAPS = IsolationCapabilities(False, False)


class TestUidAllocator:
    def test_allocates_lowest_free_pair_and_keeps_task_identity(self):
        alloc = UidAllocator(base=1000, count=8)
        assert alloc.acquire("a") == (1000, 1000)
        assert alloc.acquire("b") == (1001, 1001)
        assert alloc.acquire("a") == (1000, 1000)

    def test_exhaustion_never_falls_back_to_daemon_uid(self):
        alloc = UidAllocator(base=1000, count=1)
        alloc.acquire("a")
        with pytest.raises(ValueError, match="exhausted"):
            alloc.acquire("b")

    def test_release_does_not_recycle_possibly_live_identity(self):
        alloc = UidAllocator(base=1000, count=2)
        alloc.acquire("a")
        alloc.release("a")
        assert alloc.acquire("b") == (1001, 1001)
        assert alloc.available == 0
        with pytest.raises(ValueError, match="exhausted"):
            alloc.acquire("a")

    def test_explicit_identity_cannot_collide_with_automatic_allocation(self):
        alloc = UidAllocator(base=1000, count=2)
        alloc.reserve("explicit", 1000)
        assert alloc.acquire("automatic") == (1001, 1001)
        with pytest.raises(ValueError, match="assigned"):
            alloc.reserve("duplicate", 1000)

    def test_released_explicit_identity_cannot_be_reused(self):
        alloc = UidAllocator(base=1000, count=2)
        alloc.reserve("explicit", 2000)
        alloc.release("explicit")
        with pytest.raises(ValueError, match="assigned"):
            alloc.reserve("replacement", 2000)

    @pytest.mark.parametrize("base,count", [(0, 1), (-1, 1), (1000, 0), (MAX_UID, 2)])
    def test_rejects_privileged_and_invalid_ranges(self, base, count):
        with pytest.raises(ValueError, match="UID range"):
            UidAllocator(base=base, count=count)

    def test_release_unknown_task_is_noop(self):
        alloc = UidAllocator(base=1000, count=2)
        alloc.release("ghost")
        assert alloc.available == 2


class TestSupervisorAutoUid:
    async def test_privileged_daemon_assigns_distinct_uids(self):
        supervisor = Supervisor(capabilities=ROOT_CAPS)
        try:
            first = await supervisor.register(TaskSpec(name="a", argv=["true"]), False)
            second = await supervisor.register(TaskSpec(name="b", argv=["true"]), False)
            assert first.uid is not None
            assert second.uid is not None
            assert first.uid != second.uid
        finally:
            await supervisor.shutdown()

    @pytest.mark.parametrize(
        "caps", [USER_CAPS, IsolationCapabilities(True, False, False)]
    )
    async def test_automatic_isolation_fails_when_unavailable(self, caps):
        supervisor = Supervisor(capabilities=caps)
        with pytest.raises(IsolationError):
            await supervisor.register(TaskSpec(name="a", argv=["true"]), False)
        assert supervisor.status_all() == []

    async def test_explicit_identity_reserves_allocator_slot(self):
        supervisor = Supervisor(
            capabilities=ROOT_CAPS, uid_allocator=UidAllocator(base=1000, count=2)
        )
        try:
            explicit = await supervisor.register(
                TaskSpec(name="explicit", argv=["true"], uid=1000, gid=1000), False
            )
            automatic = await supervisor.register(
                TaskSpec(name="automatic", argv=["true"]), False
            )
            assert explicit.uid == 1000
            assert automatic.uid == 1001
            with pytest.raises(IsolationError):
                await supervisor.register(TaskSpec(name="full", argv=["true"]), False)
            assert {item.name for item in supervisor.status_all()} == {
                "explicit",
                "automatic",
            }
        finally:
            await supervisor.shutdown()

    async def test_auto_uid_opt_out_is_explicit_trusted_mode(self):
        supervisor = Supervisor(capabilities=USER_CAPS)
        try:
            status = await supervisor.register(
                TaskSpec(name="trusted", argv=["true"], auto_uid=False), False
            )
            assert status.uid is None
        finally:
            await supervisor.shutdown()

    async def test_unregister_never_reuses_uid_for_another_task(self):
        supervisor = Supervisor(capabilities=ROOT_CAPS)
        try:
            first = await supervisor.register(TaskSpec(name="a", argv=["true"]), False)
            await supervisor.unregister("a")
            second = await supervisor.register(TaskSpec(name="b", argv=["true"]), False)
            assert second.uid != first.uid
        finally:
            await supervisor.shutdown()

    async def test_uid_survives_restart_of_same_task(self, monkeypatch):
        import openenv.core.openenvd.supervisor as sup_mod

        spawned_ids = []

        async def spawn(spec, caps, *, env):
            spawned_ids.append((spec.uid, spec.gid))
            return await asyncio.create_subprocess_exec(
                *spec.argv, env=env, start_new_session=True
            )

        monkeypatch.setattr(sup_mod, "spawn_task", spawn)
        supervisor = Supervisor(capabilities=ROOT_CAPS)
        try:
            initial = await supervisor.register(
                TaskSpec(
                    name="crasher",
                    argv=["sh", "-c", "exit 1"],
                    restart_policy="on_failure",
                    max_retries=1,
                    backoff_s=0.01,
                )
            )
            for _ in range(100):
                if supervisor.status("crasher").state == TaskState.FAILED:
                    break
                await asyncio.sleep(0.02)
            assert supervisor.status("crasher").state == TaskState.FAILED
            assert spawned_ids == [(initial.uid, initial.gid)] * 2
        finally:
            await supervisor.shutdown()
