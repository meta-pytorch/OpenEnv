# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face implementation of :class:`SandboxBackend`.

Runs the OpenCode agent inside a Hugging Face sandbox (``huggingface_hub.Sandbox``,
the official SDK — the standalone ``hf_sandbox`` package was an intermediate step now
folded into ``huggingface_hub``, available since ``huggingface_hub>=1.22``).

The agent image must ship the ``opencode`` CLI, node, and the in-sandbox proxy
(the E2B path bakes these via ``build_template.py``; the HF path needs an equivalent
image passed as ``image=``).
"""

from __future__ import annotations

import time
from pathlib import PurePosixPath

from huggingface_hub import Sandbox

from .base import BgJob, ExecResult, SandboxBackend, SandboxHandle


_WAIT_POLL_INTERVAL_S = 0.5


class HFBgJob:
    """Wraps a ``huggingface_hub`` ``SandboxProcess`` to satisfy :class:`BgJob`.

    The SDK's ``SandboxProcess`` is a snapshot with no blocking ``wait()``, so we
    poll ``Sandbox.processes`` for our pid and raise ``TimeoutError`` if the process
    does not exit within the caller-supplied budget. Sandbox lifetime still bounds it.
    """

    def __init__(self, sandbox: Sandbox, process) -> None:
        self._sandbox = sandbox
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            proc = next((p for p in self._sandbox.processes() if p.pid == self.pid), None)
            # Completed processes stay listed with running=False; a missing pid means
            # it exited and was reaped, so treat both as "done".
            if proc is None:
                return 0
            if not proc.running:
                return int(proc.exit_code) if proc.exit_code is not None else 0
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Background command did not exit within {timeout}s")
            time.sleep(_WAIT_POLL_INTERVAL_S)

    def kill(self) -> None:
        try:
            self._process.kill()  # idempotent server-side
        except Exception:
            pass


class HFSandboxHandle:
    """Wraps a live ``huggingface_hub.Sandbox`` to satisfy :class:`SandboxHandle`."""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sbx = sandbox

    @property
    def sandbox_id(self) -> str:
        return self._sbx.id

    @property
    def raw(self) -> Sandbox:
        """Escape hatch for callers that need the underlying SDK object."""
        return self._sbx

    def exec(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> ExecResult:
        # check=False: non-zero exits are expected in many contexts (e.g. polling
        # healthz before the server is up), so return them as an ExecResult instead
        # of raising. Run through a shell so `cmd` is a single command string.
        result = self._sbx.run(
            ["bash", "-lc", cmd],
            env=envs,
            cwd=cwd,
            timeout=timeout,
            check=False,
        )
        return ExecResult(
            exit_code=int(result.exit_code) if result.exit_code is not None else 0,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def start_bg(
        self,
        cmd: str,
        *,
        envs: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> BgJob:
        # Detached: the streaming/timeout/check options do not apply in background
        # mode. Sandbox lifetime bounds the process.
        process = self._sbx.run(
            ["bash", "-lc", cmd],
            env=envs,
            cwd=cwd,
            background=True,
        )
        return HFBgJob(self._sbx, process)

    def write_text(self, path: str, content: str) -> None:
        parent = str(PurePosixPath(path).parent)
        if parent not in ("", "/"):
            self._sbx.files.mkdir(parent)
        self._sbx.files.write(path, content)

    def read_text(self, path: str) -> str:
        return self._sbx.files.read_text(path)

    def exists(self, path: str) -> bool:
        return self._sbx.files.exists(path)

    def kill(self) -> None:
        # `kill()` terminates the sandbox (and its backing job); `close()` would only
        # release the local HTTP client and leave the sandbox idling until its timeout.
        self._sbx.kill()


class HFSandboxBackend:
    """Creates Hugging Face sandboxes for OpenCode rollouts.

    ``image`` must contain the ``opencode`` CLI, node, and the in-sandbox proxy.
    Extra ``huggingface_hub.Sandbox.create`` options can be forwarded via
    ``sandbox_kwargs``.
    """

    def __init__(
        self,
        *,
        image: str,
        flavor: str = "cpu-basic",
        forward_hf_token: bool = False,
        sandbox_kwargs: dict | None = None,
    ) -> None:
        self._image = image
        self._flavor = flavor
        self._forward_hf_token = forward_hf_token
        self._sandbox_kwargs = sandbox_kwargs or {}

    def create(
        self,
        *,
        timeout_s: int = 900,
        envs: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SandboxHandle:
        # NOTE: huggingface_hub.Sandbox.create has no `metadata` parameter (unlike
        # E2B), so `metadata` is accepted for protocol parity but not forwarded.
        sbx = Sandbox.create(
            image=self._image,
            flavor=self._flavor,
            idle_timeout=timeout_s,
            env=envs,
            forward_hf_token=self._forward_hf_token,
            **self._sandbox_kwargs,
        )
        return HFSandboxHandle(sbx)
