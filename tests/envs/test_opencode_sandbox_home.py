# SPDX-License-Identifier: BSD-3-Clause

"""The opencode_runtime path helpers derive from ``config.sandbox_home``.

Regression test for the fix that lets root-based sandbox backends (HF, Docker) work:
the proxy/opencode paths must follow ``sandbox_home`` instead of a hardcoded ``/home/user``.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opencode_env.config import OpenCodeConfig  # noqa: E402
from opencode_env.opencode_runtime import (  # noqa: E402
    agent_log_path,
    opencode_bin_path,
    opencode_config_path,
    proxy_dir,
    proxy_log_path,
    proxy_source_path,
    proxy_trace_path,
    workdir_path,
)


def _cfg(sandbox_home=None):
    kwargs = {"base_url": "http://localhost:8000/v1"}
    if sandbox_home is not None:
        kwargs["sandbox_home"] = sandbox_home
    return OpenCodeConfig(**kwargs)


def test_default_home_is_home_user():
    c = _cfg()  # default sandbox_home == /home/user (E2B)
    assert proxy_trace_path(c) == "/home/user/logs/agent/proxy_trace.jsonl"
    assert proxy_log_path(c) == "/home/user/logs/agent/proxy.log"
    assert proxy_dir(c) == "/home/user/proxy"
    assert proxy_source_path(c) == "/home/user/proxy/interception.py"
    assert opencode_bin_path(c) == "/home/user/.opencode/bin/opencode"


def test_root_sandbox_home():
    c = _cfg("/root")  # HF sandbox / Docker exec as root
    assert proxy_trace_path(c) == "/root/logs/agent/proxy_trace.jsonl"
    assert proxy_log_path(c) == "/root/logs/agent/proxy.log"
    assert proxy_dir(c) == "/root/proxy"
    assert proxy_source_path(c) == "/root/proxy/interception.py"
    assert opencode_bin_path(c) == "/root/.opencode/bin/opencode"


def test_every_path_helper_sits_under_sandbox_home():
    c = _cfg("/custom/home")
    for path in (
        proxy_trace_path(c),
        proxy_log_path(c),
        proxy_dir(c),
        proxy_source_path(c),
        opencode_bin_path(c),
        agent_log_path(c),
        opencode_config_path(c),
        workdir_path(c),
    ):
        assert path.startswith("/custom/home/"), path
