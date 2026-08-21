# SPDX-License-Identifier: BSD-3-Clause

"""Tests for openenvd isolation primitives (uid drop, network namespace)."""

import os

import pytest
from openenv.core.openenvd import isolation
from openenv.core.openenvd.models import TaskSpec


@pytest.fixture
def root_caps():
    return isolation.IsolationCapabilities(can_drop_uid=True, can_unshare_net=True)


@pytest.fixture
def unprivileged_caps():
    return isolation.IsolationCapabilities(can_drop_uid=False, can_unshare_net=False)


class TestDetectCapabilities:
    def test_returns_capabilities(self):
        caps = isolation.detect_capabilities()
        assert isinstance(caps.can_drop_uid, bool)
        assert isinstance(caps.can_unshare_net, bool)

    def test_matches_euid_for_uid_drop(self):
        caps = isolation.detect_capabilities()
        assert caps.can_drop_uid == (os.geteuid() == 0)


class TestBuildPreexec:
    def test_no_isolation_returns_none(self, root_caps):
        spec = TaskSpec(name="x", argv=["true"])
        preexec, warnings = isolation.build_preexec(spec, root_caps)
        assert preexec is None
        assert warnings == []

    def test_network_isolation_without_capability_warns(self, unprivileged_caps):
        spec = TaskSpec(name="x", argv=["true"], network_isolated=True)
        preexec, warnings = isolation.build_preexec(spec, unprivileged_caps)
        assert preexec is None
        assert any("network" in w.lower() for w in warnings)

    def test_uid_drop_without_capability_warns(self, unprivileged_caps):
        spec = TaskSpec(name="x", argv=["true"], uid=1000, gid=1000)
        preexec, warnings = isolation.build_preexec(spec, unprivileged_caps)
        assert preexec is None
        assert any("uid" in w.lower() for w in warnings)

    def test_network_isolated_preexec_unshares(self, root_caps, monkeypatch):
        calls = []
        monkeypatch.setattr(
            isolation, "_unshare_net", lambda: calls.append("unshare") or True
        )
        spec = TaskSpec(name="x", argv=["true"], network_isolated=True)
        preexec, warnings = isolation.build_preexec(spec, root_caps)
        assert preexec is not None
        assert warnings == []
        preexec()
        assert calls == ["unshare"]

    def test_uid_drop_preexec_sets_ids(self, root_caps, monkeypatch):
        calls = []
        monkeypatch.setattr(os, "setgid", lambda g: calls.append(("setgid", g)))
        monkeypatch.setattr(os, "setuid", lambda u: calls.append(("setuid", u)))
        monkeypatch.setattr(
            os, "setgroups", lambda groups: calls.append(("setgroups", groups))
        )
        spec = TaskSpec(name="x", argv=["true"], uid=1234, gid=5678)
        preexec, warnings = isolation.build_preexec(spec, root_caps)
        assert preexec is not None
        assert warnings == []
        preexec()
        assert ("setgroups", []) in calls
        assert ("setgid", 5678) in calls
        assert ("setuid", 1234) in calls
        kinds = [c[0] for c in calls]
        assert kinds.index("setgid") < kinds.index("setuid")

    def test_combined_isolation(self, root_caps, monkeypatch):
        calls = []
        monkeypatch.setattr(
            isolation, "_unshare_net", lambda: calls.append("unshare") or True
        )
        monkeypatch.setattr(os, "setgid", lambda g: None)
        monkeypatch.setattr(os, "setuid", lambda u: None)
        monkeypatch.setattr(os, "setgroups", lambda groups: None)
        spec = TaskSpec(
            name="x", argv=["true"], uid=1000, gid=1000, network_isolated=True
        )
        preexec, _ = isolation.build_preexec(spec, root_caps)
        preexec()
        assert "unshare" in calls


def _read_net_ns_link(pid=None):
    pid = pid or os.getpid()
    return os.readlink(f"/proc/{pid}/ns/net")


def _in_initial_user_ns() -> bool:
    try:
        with open("/proc/self/uid_map") as f:
            first = f.readline().split()
    except OSError:
        return False
    return first == ["0", "0", "4294967295"]


requires_root = pytest.mark.skipif(os.geteuid() != 0, reason="requires root")
requires_real_root = pytest.mark.skipif(
    os.geteuid() != 0 or not _in_initial_user_ns(),
    reason="requires real root (initial user ns) for setuid to arbitrary uids",
)


class TestRealIsolation:
    @requires_root
    async def test_network_isolated_child_gets_new_netns(self):
        import asyncio

        parent_ns = _read_net_ns_link()
        spec = TaskSpec(name="x", argv=["sleep", "5"], network_isolated=True)
        preexec, warnings = isolation.build_preexec(spec)
        assert preexec is not None
        assert warnings == []
        proc = await asyncio.create_subprocess_exec(
            "sleep",
            "5",
            preexec_fn=preexec,
            start_new_session=True,
        )
        try:
            child_ns = _read_net_ns_link(proc.pid)
            assert child_ns != parent_ns
        finally:
            proc.kill()
            await proc.wait()

    @requires_real_root
    async def test_uid_dropped_child_runs_as_target_uid(self):
        import asyncio

        target_uid = 65534
        target_gid = 65534
        marker = "/tmp/openenvd-uid-test"
        if os.path.exists(marker):
            os.unlink(marker)
        script = f"id -u > {marker}"
        spec = TaskSpec(name="x", argv=["sh"], uid=target_uid, gid=target_gid)
        preexec, warnings = isolation.build_preexec(spec)
        assert preexec is not None
        assert warnings == []
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            script,
            preexec_fn=preexec,
        )
        await proc.wait()
        assert os.path.exists(marker), "child did not run"
        with open(marker) as f:
            assert int(f.read().strip()) == target_uid
        os.unlink(marker)
