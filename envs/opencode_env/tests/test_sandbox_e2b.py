# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Live E2B integration tests — skipped unless E2B_API_KEY is set.

Run with: ``PYTHONPATH=src:envs/opencode_env E2B_API_KEY=... uv run pytest
envs/opencode_env/tests/test_sandbox_e2b.py -v``
"""

from __future__ import annotations

import os

import pytest

from opencode_env.config import OpenCodeConfig
from opencode_env.opencode_runtime import build_install_cmd
from opencode_env.sandbox import E2BSandboxBackend

pytestmark = pytest.mark.skipif(
    not os.environ.get("E2B_API_KEY"),
    reason="E2B_API_KEY not set; skipping live E2B tests",
)


@pytest.fixture
def sandbox():
    backend = E2BSandboxBackend()
    sbx = backend.create(timeout_s=300)
    yield sbx
    sbx.kill()


def test_sandbox_echo(sandbox):
    r = sandbox.exec("echo hello")
    assert r.exit_code == 0
    assert "hello" in r.stdout


def test_sandbox_write_and_read(sandbox):
    sandbox.write_text("/task/instruction.md", "solve fizzbuzz")
    assert sandbox.exists("/task/instruction.md")
    assert sandbox.read_text("/task/instruction.md") == "solve fizzbuzz"


def test_install_opencode(sandbox):
    cfg = OpenCodeConfig(base_url="http://localhost:0/v1")  # unused here
    r = sandbox.exec(build_install_cmd(cfg), timeout=180)
    assert r.exit_code == 0, f"install failed: {r.stderr}"
    # opencode --version is the last command in the install script
    assert "opencode" in (r.stdout + r.stderr).lower()
