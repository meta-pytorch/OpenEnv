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


def test_wait_returns_zero_when_pid_disappears():
    # Process reaped and gone from the listing -> done, exit 0.
    proc = _FakeProcess(pid=7, running=True)
    assert HFBgJob(_FakeSandbox([]), proc).wait(timeout=1.0) == 0


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
