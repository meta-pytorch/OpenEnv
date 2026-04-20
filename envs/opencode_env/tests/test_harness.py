# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for OpenCodeSession / OpenCodeSessionFactory (no sandbox)."""

from __future__ import annotations

import pytest

from opencode_env.config import OpenCodeConfig
from opencode_env.harness import OpenCodeSession, OpenCodeSessionFactory
from opencode_env.sandbox.base import ExecResult
from opencode_env.task import OpenCodeTask


class _FakeBgJob:
    def __init__(self) -> None:
        self.pid = 123
        self._killed = False

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self._killed = True


class _FakeSandbox:
    """In-memory sandbox that records every interaction."""

    def __init__(self, *, install_exit: int = 0, setup_exit: int = 0) -> None:
        self.sandbox_id = "fake-sbx"
        self.exec_calls: list[tuple[str, dict | None]] = []
        self.written: dict[str, str] = {}
        self.bg_calls: list[tuple[str, dict | None]] = []
        self.killed = False
        self._install_exit = install_exit
        self._setup_exit = setup_exit

    def exec(self, cmd, *, envs=None, cwd=None, timeout=60):
        self.exec_calls.append((cmd, envs))
        if "opencode.ai/install" in cmd:
            return ExecResult(self._install_exit, "opencode 0.0.0\n", "")
        return ExecResult(self._setup_exit, "", "")

    def start_bg(self, cmd, *, envs=None, cwd=None):
        self.bg_calls.append((cmd, envs))
        return _FakeBgJob()

    def write_text(self, path, content):
        self.written[path] = content

    def read_text(self, path):
        return self.written.get(path, "")

    def exists(self, path):
        return path in self.written

    def kill(self):
        self.killed = True


class _FakeBackend:
    def __init__(self, sandbox: _FakeSandbox) -> None:
        self._sandbox = sandbox
        self.create_calls = 0

    def create(self, *, timeout_s=900, envs=None, metadata=None):
        self.create_calls += 1
        return self._sandbox


def _config(**overrides) -> OpenCodeConfig:
    base = dict(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_key="sk-fake",
        model="openai/gpt-5.3-codex",
    )
    base.update(overrides)
    return OpenCodeConfig(**base)


def test_factory_bootstraps_and_starts_agent():
    sbx = _FakeSandbox()
    backend = _FakeBackend(sbx)
    factory = OpenCodeSessionFactory(config=_config(), sandbox_backend=backend)

    session = factory.create(task="solve fizzbuzz")

    assert backend.create_calls == 1
    assert any("opencode.ai/install" in c for c, _ in sbx.exec_calls)
    assert "/home/user/.config/opencode/opencode.json" in sbx.written
    assert sbx.written["/home/user/task/instruction.md"] == "solve fizzbuzz"
    assert len(sbx.bg_calls) == 1, "agent must be started in background"
    # OPENAI_BASE_URL must be injected into the process env
    _, envs = sbx.bg_calls[0]
    assert envs["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert envs["OPENAI_API_KEY"] == "sk-fake"
    assert isinstance(session, OpenCodeSession)


def test_factory_runs_task_setup_shell():
    sbx = _FakeSandbox()
    factory = OpenCodeSessionFactory(
        config=_config(), sandbox_backend=_FakeBackend(sbx)
    )
    task = OpenCodeTask(instruction="x", setup_shell="pip install pytest")

    factory.create(task=task)

    setup_cmds = [c for c, _ in sbx.exec_calls if "pip install" in c]
    assert setup_cmds == ["pip install pytest"]


def test_factory_uploads_extra_files():
    sbx = _FakeSandbox()
    factory = OpenCodeSessionFactory(
        config=_config(), sandbox_backend=_FakeBackend(sbx)
    )
    task = OpenCodeTask(
        instruction="run it",
        upload_files={"/home/user/workdir/hello.py": "print('hi')"},
    )

    factory.create(task=task)

    assert sbx.written["/home/user/workdir/hello.py"] == "print('hi')"


def test_factory_kills_sandbox_on_install_failure():
    sbx = _FakeSandbox(install_exit=1)
    factory = OpenCodeSessionFactory(
        config=_config(), sandbox_backend=_FakeBackend(sbx)
    )

    with pytest.raises(RuntimeError, match="install failed"):
        factory.create(task="x")
    assert sbx.killed


def test_factory_accepts_transparent_proxy_mode():
    f = OpenCodeSessionFactory(
        config=_config(),
        sandbox_backend=_FakeBackend(_FakeSandbox()),
        mode="transparent_proxy",
    )
    assert f._mode == "transparent_proxy"


def test_factory_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        OpenCodeSessionFactory(
            config=_config(),
            sandbox_backend=_FakeBackend(_FakeSandbox()),
            mode="bogus",  # type: ignore[arg-type]
        )


def test_session_initial_messages():
    sbx = _FakeSandbox()
    session = OpenCodeSession(
        sandbox=sbx,
        config=_config(),
        task=OpenCodeTask(instruction="hi"),
    )
    assert session.initial_messages() == [{"role": "user", "content": "hi"}]


def test_session_verify_without_verifier_returns_none_reward():
    sbx = _FakeSandbox()
    session = OpenCodeSession(
        sandbox=sbx,
        config=_config(),
        task=OpenCodeTask(instruction="x"),
    )
    result = session.verify(transcript=[])
    assert result.env_reward is None
    assert result.done is True


def test_session_verify_calls_user_verifier():
    from openenv.core.harness import VerifyResult

    sbx = _FakeSandbox()
    calls = []

    def verifier(sandbox, task):
        calls.append((sandbox.sandbox_id, task.instruction))
        return VerifyResult(env_reward=1.0, done=True, metrics={"tests": "pass"})

    session = OpenCodeSession(
        sandbox=sbx,
        config=_config(),
        task=OpenCodeTask(instruction="do"),
        verifier=verifier,
    )
    result = session.verify(transcript=[])
    assert calls == [("fake-sbx", "do")]
    assert result.env_reward == 1.0
    assert result.metrics == {"tests": "pass"}


def test_session_close_kills_job_and_sandbox():
    sbx = _FakeSandbox()
    session = OpenCodeSession(
        sandbox=sbx,
        config=_config(),
        task=OpenCodeTask(instruction="x"),
    )
    session._bg_job = _FakeBgJob()
    session.close()
    assert session._bg_job is None
    assert sbx.killed
