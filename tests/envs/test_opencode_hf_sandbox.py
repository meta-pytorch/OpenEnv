# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``HFBgJob.wait()`` polling.

Pure logic, no network and no real Hugging Face sandbox: ``HFBgJob`` takes a
sandbox object and a process object, so both are faked here. ``hf.py`` does
``from huggingface_hub import Sandbox`` at import time; if this ``huggingface_hub``
predates ``Sandbox`` (added in ``>=1.22``) we stub the attribute so the pure
``HFBgJob`` logic is still testable.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make ``envs/`` importable when running from the repository root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import huggingface_hub  # noqa: E402

if not hasattr(huggingface_hub, "Sandbox"):  # pragma: no cover - only on old hub
    huggingface_hub.Sandbox = type("Sandbox", (), {})  # type: ignore[attr-defined]

import opencode_env.sandbox.hf as hf_mod  # noqa: E402
from opencode_env.sandbox.hf import HFBgJob  # noqa: E402


class _FakeProcess:
    def __init__(
        self, pid: int, running: bool = True, exit_code: int | None = None
    ) -> None:
        self.pid = pid
        self.running = running
        self.exit_code = exit_code
        self.killed = False

    def kill(self) -> None:
        self.killed = True


class _FakeSandbox:
    """``.processes()`` returns the current snapshot list (a method, matching the real SDK)."""

    def __init__(self, procs: list) -> None:
        self._procs = procs

    def processes(self) -> list:
        return list(self._procs)


class _FlippingSandbox:
    """Reports the process as running for ``flip_after`` polls, then exited."""

    def __init__(self, proc: _FakeProcess, *, flip_after: int, exit_code: int) -> None:
        self._proc = proc
        self._flip_after = flip_after
        self._exit_code = exit_code
        self.poll_count = 0

    def processes(self) -> list:
        self.poll_count += 1
        if self.poll_count >= self._flip_after:
            self._proc.running = False
            self._proc.exit_code = self._exit_code
        return [self._proc]


@pytest.fixture(autouse=True)
def _instant_poll(monkeypatch):
    # Keep the polling loop instant so timeout tests stay sub-second.
    monkeypatch.setattr(hf_mod, "_WAIT_POLL_INTERVAL_S", 0.0)


def test_pid_property():
    proc = _FakeProcess(pid=42)
    assert HFBgJob(_FakeSandbox([proc]), proc).pid == 42


def test_wait_returns_zero_when_already_exited():
    proc = _FakeProcess(pid=7, running=False, exit_code=0)
    assert HFBgJob(_FakeSandbox([proc]), proc).wait(timeout=1.0) == 0


def test_wait_returns_nonzero_exit_code():
    proc = _FakeProcess(pid=7, running=False, exit_code=3)
    assert HFBgJob(_FakeSandbox([proc]), proc).wait(timeout=1.0) == 3


def test_wait_returns_zero_when_exit_code_unknown():
    # running=False but exit_code never populated -> treated as 0.
    proc = _FakeProcess(pid=7, running=False, exit_code=None)
    assert HFBgJob(_FakeSandbox([proc]), proc).wait(timeout=1.0) == 0


def test_wait_polls_until_process_exits():
    proc = _FakeProcess(pid=7, running=True)
    sbx = _FlippingSandbox(proc, flip_after=3, exit_code=5)
    assert HFBgJob(sbx, proc).wait(timeout=1.0) == 5
    assert sbx.poll_count >= 3  # it actually polled, didn't short-circuit


def test_wait_raises_when_pid_disappears():
    # A vanished pid means the sandbox was torn down mid-run, not a clean exit.
    proc = _FakeProcess(pid=7, running=True)
    with pytest.raises(RuntimeError, match="vanished"):
        HFBgJob(_FakeSandbox([]), proc).wait(timeout=1.0)


def test_wait_raises_timeout_when_never_exits():
    proc = _FakeProcess(pid=7, running=True)
    job = HFBgJob(_FakeSandbox([proc]), proc)  # always running
    with pytest.raises(TimeoutError):
        job.wait(timeout=0.02)


def test_wait_none_timeout_polls_without_deadline(monkeypatch):
    # timeout=None must not raise; it should keep polling until exit.
    proc = _FakeProcess(pid=7, running=True)
    sbx = _FlippingSandbox(proc, flip_after=2, exit_code=0)
    assert HFBgJob(sbx, proc).wait(timeout=None) == 0


def test_kill_calls_through():
    proc = _FakeProcess(pid=7)
    HFBgJob(_FakeSandbox([proc]), proc).kill()
    assert proc.killed is True


def test_kill_swallows_errors():
    class _RaisingProc(_FakeProcess):
        def kill(self) -> None:
            raise RuntimeError("boom")

    proc = _RaisingProc(pid=7)
    HFBgJob(_FakeSandbox([proc]), proc).kill()  # must not raise


# --------------------------------------------------------------------------
# HFSandboxHandle / HFSandboxBackend
# --------------------------------------------------------------------------

from opencode_env.sandbox.hf import HFSandboxBackend, HFSandboxHandle  # noqa: E402


class _FakeCmdResult:
    def __init__(self, exit_code, stdout="", stderr="", timed_out=False):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


class _FakeFiles:
    def __init__(self):
        self.written = {}
        self.made = []

    def mkdir(self, path):
        self.made.append(path)

    def write(self, path, content):
        self.written[path] = content

    def read_text(self, path):
        return self.written[path]

    def exists(self, path):
        return path in self.written


class _FakeRealSandbox:
    """Stands in for ``huggingface_hub.Sandbox`` (run/files/processes/kill/id)."""

    def __init__(self, result=None):
        self.id = "sbx-123"
        self.files = _FakeFiles()
        self.killed = False
        self.calls = []
        self._result = result or _FakeCmdResult(0, stdout="ok")

    def run(self, argv, env=None, cwd=None, timeout=None, check=None, background=False):
        self.calls.append(
            {"argv": argv, "env": env, "cwd": cwd, "background": background}
        )
        if background:
            return _FakeProcess(pid=101)
        return self._result

    def processes(self):
        return []

    def kill(self):
        self.killed = True


def test_exec_maps_result():
    sbx = _FakeRealSandbox(_FakeCmdResult(0, stdout="hi", stderr=""))
    r = HFSandboxHandle(sbx).exec("echo hi", timeout=5)
    assert (r.exit_code, r.stdout) == (0, "hi")
    assert sbx.calls[0]["argv"] == ["bash", "-lc", "echo hi"]


def test_exec_timeout_surfaces_as_nonzero():
    sbx = _FakeRealSandbox(_FakeCmdResult(None, timed_out=True))
    r = HFSandboxHandle(sbx).exec("sleep 99", timeout=5)
    assert r.exit_code == 124
    assert "timed out" in r.stderr


def test_exec_none_exit_code_defaults_zero():
    r = HFSandboxHandle(_FakeRealSandbox(_FakeCmdResult(None))).exec("x")
    assert r.exit_code == 0


def test_start_bg_returns_bgjob():
    job = HFSandboxHandle(_FakeRealSandbox()).start_bg("sleep 1")
    assert isinstance(job, HFBgJob)
    assert job.pid == 101


def test_write_text_creates_parent_then_writes():
    sbx = _FakeRealSandbox()
    HFSandboxHandle(sbx).write_text("/root/a/b.txt", "data")
    assert sbx.files.made == ["/root/a"]
    assert sbx.files.written["/root/a/b.txt"] == "data"


def test_write_text_skips_mkdir_for_rootless_paths():
    sbx = _FakeRealSandbox()
    HFSandboxHandle(sbx).write_text("file.txt", "x")  # parent == "."
    assert sbx.files.made == []


def test_read_text_and_exists():
    h = HFSandboxHandle(_FakeRealSandbox())
    h.write_text("/root/f", "v")
    assert h.read_text("/root/f") == "v"
    assert h.exists("/root/f") is True
    assert h.exists("/root/missing") is False


def test_kill_tears_down_sandbox():
    sbx = _FakeRealSandbox()
    HFSandboxHandle(sbx).kill()
    assert sbx.killed is True


def test_sandbox_id_uses_dot_id():
    assert HFSandboxHandle(_FakeRealSandbox()).sandbox_id == "sbx-123"


def test_backend_create_forwards_kwargs(monkeypatch):
    captured = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeRealSandbox()

    monkeypatch.setattr(
        hf_mod.Sandbox, "create", staticmethod(_fake_create), raising=False
    )
    backend = HFSandboxBackend(
        image="python:3.12", flavor="cpu-basic", forward_hf_token=True
    )
    handle = backend.create(timeout_s=600)
    assert isinstance(handle, HFSandboxHandle)
    assert captured["image"] == "python:3.12"
    assert captured["flavor"] == "cpu-basic"
    assert captured["idle_timeout"] == 600
    assert captured["forward_hf_token"] is True
