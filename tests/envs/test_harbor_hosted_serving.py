# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serving on a hosted platform, where there is one port and one URL.

Locally the capture proxy runs on its own port and is published to the sandbox. A Space exposes
exactly one port and already has a public URL, so the proxy is mounted onto the env server's app
instead and nothing is forwarded. These tests pin that split, because getting it wrong is not a
crash: it is a deployment that quietly opens a second listener it cannot publish.
"""

from __future__ import annotations

import pytest

serving = pytest.importorskip("openenv.harbor.serving")

CAPTURE_MOUNT = serving.CAPTURE_MOUNT
HarborService = serving.HarborService
space_public_url = serving.space_public_url

# Never contacted: nothing here reaches an engine.
UNUSED_LLM = "http://127.0.0.1:9/v1"


@pytest.fixture(autouse=True)
def _clear_space_env(monkeypatch):
    """Tests must not inherit a Space identity from the developer's shell."""
    monkeypatch.delenv("SPACE_HOST", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)


def test_no_space_means_no_public_url():
    assert space_public_url() == ""


def test_space_host_is_used_verbatim(monkeypatch):
    monkeypatch.setenv("SPACE_HOST", "owner-env.hf.space")
    assert space_public_url() == "https://owner-env.hf.space"


def test_space_host_tolerates_a_scheme_already_present(monkeypatch):
    monkeypatch.setenv("SPACE_HOST", "https://owner-env.hf.space/")
    assert space_public_url() == "https://owner-env.hf.space"


def test_space_id_is_slugged_when_host_is_absent(monkeypatch):
    """`SPACE_ID` is always set; the hostname lowercases and dash-separates it."""
    monkeypatch.setenv("SPACE_ID", "AdithyaSK/harbor_data.agent-env")
    assert space_public_url() == "https://adithyask-harbor-data-agent-env.hf.space"


def test_hosted_start_mounts_and_never_forwards(monkeypatch):
    """The regression that got a Space flagged: a hosted deployment must not forward."""
    monkeypatch.setenv("SPACE_ID", "owner/env")

    def explode(*_args, **_kwargs):
        raise AssertionError("a hosted deployment must not create a forwarder")

    monkeypatch.setattr(
        "openenv.core.harness.capture.forwarding.make_forwarder", explode, raising=False
    )

    service = HarborService(llm_url=UNUSED_LLM, model="m", datasets=[])
    url = service.start()

    assert service.mounted is True
    assert url == f"https://owner-env.hf.space{CAPTURE_MOUNT}"
    # No port was bound, so stop() must be safe even though start() never launched a server.
    service.stop()


def test_mounted_capture_answers_under_the_prefix():
    """Mounting strips the prefix, so every dialect route keeps working unchanged."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openenv.core.harness.capture.server import create_app

    host = fastapi.FastAPI()
    host.mount(CAPTURE_MOUNT, create_app(llm_url=UNUSED_LLM, model="m"))
    client = TestClient(host)

    assert client.get(f"{CAPTURE_MOUNT}/health").json()["status"] == "ok"

    # The proxy's catch-all must match /v1/chat/completions, not /capture/v1/chat/completions.
    response = client.post(
        f"{CAPTURE_MOUNT}/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer not-a-session"},
    )
    # 401 rather than 404 proves it routed to the proxy and was rejected on identity, which is also
    # what stops a publicly mounted proxy being an open relay.
    assert response.status_code == 401
    assert "unknown API key" in response.json()["error"]["message"]
