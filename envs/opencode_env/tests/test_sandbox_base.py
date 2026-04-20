# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Structural tests for the sandbox Protocols (no live sandbox needed)."""

from __future__ import annotations

from opencode_env.sandbox import (
    ExecResult,
    SandboxBackend,
    SandboxHandle,
    E2BSandboxBackend,
    E2BSandboxHandle,
)


def test_e2b_classes_import():
    # Ensure the e2b backend imports without needing an API key or live call.
    assert E2BSandboxBackend is not None
    assert E2BSandboxHandle is not None


def test_exec_result_dataclass():
    r = ExecResult(exit_code=0, stdout="ok", stderr="")
    assert r.exit_code == 0
    assert r.stdout == "ok"


def test_e2b_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="E2B_API_KEY"):
        E2BSandboxBackend()


def test_protocols_are_declared():
    # Static: the Protocols should be importable and non-empty.
    assert hasattr(SandboxBackend, "create")
    assert hasattr(SandboxHandle, "exec")
    assert hasattr(SandboxHandle, "start_bg")
    assert hasattr(SandboxHandle, "kill")
