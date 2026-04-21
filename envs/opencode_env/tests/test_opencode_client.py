# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for :class:`OpenCodeServerClient`.

No live opencode serve needed — uses ``httpx.MockTransport`` to assert the
wire contract (URLs, methods, body shapes, auth header).
"""

from __future__ import annotations

import json

import httpx
import pytest

from opencode_env.opencode_client import (
    OpenCodeServerClient,
    event_part,
    event_session_id,
)


def _transport(handler):
    """Return an httpx transport that routes every request through ``handler``."""
    return httpx.MockTransport(handler)


def test_create_session_request_shape(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        seen["content_type"] = request.headers.get("content-type")
        return httpx.Response(200, json={"id": "ses_x", "title": "t"})

    _patch_httpx(monkeypatch, handler)
    client = OpenCodeServerClient("http://server:4096")
    out = client.create_session(title="spike")
    assert out == {"id": "ses_x", "title": "t"}
    assert seen["url"] == "http://server:4096/session"
    assert seen["method"] == "POST"
    assert seen["body"] == {"title": "spike"}
    assert seen["content_type"] == "application/json"


def test_auth_header_applied_when_password_set(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    _patch_httpx(monkeypatch, handler)
    client = OpenCodeServerClient("http://server:4096", password="hunter2")
    client.get_session("s")
    # "opencode:hunter2" base64-encoded
    assert seen["auth"] == "Basic b3BlbmNvZGU6aHVudGVyMg=="


def test_send_message_includes_text_part(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"info": {"id": "msg_1"}, "parts": [{"type": "text", "text": "done"}]},
        )

    _patch_httpx(monkeypatch, handler)
    client = OpenCodeServerClient("http://server:4096")
    out = client.send_message("ses_x", "hello", model="openai_compatible/Qwen")
    assert "hello" in captured["body"]["parts"][0]["text"]
    assert captured["body"]["model"] == "openai_compatible/Qwen"
    assert captured["url"].endswith("/session/ses_x/message")
    assert out["info"]["id"] == "msg_1"


def test_abort_returns_bool(monkeypatch):
    _patch_httpx(monkeypatch, lambda r: httpx.Response(200, json=True))
    client = OpenCodeServerClient("http://server:4096")
    assert client.abort("ses_x") is True


def test_list_messages_passes_limit(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    _patch_httpx(monkeypatch, handler)
    client = OpenCodeServerClient("http://server:4096")
    client.list_messages("s", limit=25)
    assert seen["query"] == {"limit": "25"}


def test_wait_for_ready_polls(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        status = 200 if calls["n"] >= 3 else 503
        return httpx.Response(status, text="{}")

    _patch_httpx(monkeypatch, handler)
    client = OpenCodeServerClient("http://server:4096")
    # Use tiny interval so the test is fast.
    assert client.wait_for_ready(attempts=10, interval_s=0.01) is True
    assert calls["n"] == 3


def test_event_part_and_session_id_extractors():
    ev = {
        "type": "message.part.updated",
        "properties": {
            "info": {"id": "msg_x", "sessionID": "ses_x"},
            "part": {"type": "text", "text": "hello"},
        },
    }
    assert event_session_id(ev) == "ses_x"
    assert event_part(ev) == {"type": "text", "text": "hello"}

    # Defensive: empty event
    assert event_session_id({}) is None
    assert event_part({}) == {}


# ---------------------------------------------------------------------------
# monkeypatch helper — intercept every httpx.get/httpx.post made by the client
# so we don't rely on per-call transport parameters.
# ---------------------------------------------------------------------------


def _patch_httpx(monkeypatch, handler):
    import httpx as _httpx

    client = _httpx.Client(transport=_transport(handler))

    def _get(url, *a, **kw):
        return client.get(url, *a, **kw)

    def _post(url, *a, **kw):
        return client.post(url, *a, **kw)

    monkeypatch.setattr(_httpx, "get", _get)
    monkeypatch.setattr(_httpx, "post", _post)
