# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the interception proxy (no sandbox, no real LLM)."""

from __future__ import annotations

import json
import os
import socket
from contextlib import closing

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request

from opencode_env.interception import (
    InterceptionProxy,
    ProxyConfig,
    _build_turn_record,
    _strip_logprobs,
    read_trace,
)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_upstream_app(response_payload: dict) -> FastAPI:
    app = FastAPI()
    received: list[dict] = []

    @app.post("/v1/chat/completions")
    async def handler(request: Request):
        body = await request.json()
        received.append(body)
        return response_payload

    app.state.received = received
    return app


def _run_upstream(app: FastAPI, port: int) -> uvicorn.Server:
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    import threading

    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    import time

    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    return server
        except OSError:
            pass
        time.sleep(0.05)
    raise RuntimeError("upstream failed to start")


_FAKE_RESPONSE = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
            "logprobs": {
                "content": [
                    {"token": "h", "logprob": -0.1, "top_logprobs": []},
                    {"token": "i", "logprob": -0.2, "top_logprobs": []},
                ]
            },
        }
    ],
}


def test_strip_logprobs_removes_only_logprobs_key():
    sanitized = _strip_logprobs(_FAKE_RESPONSE)
    choice = sanitized["choices"][0]
    assert "logprobs" not in choice
    assert choice["message"]["content"] == "hi"
    assert choice["finish_reason"] == "stop"


def test_build_turn_record_extracts_logprobs():
    record = _build_turn_record(
        turn_idx=1,
        request_body={"model": "test", "messages": []},
        response_json=_FAKE_RESPONSE,
        latency_s=0.25,
    )
    assert record.completion_tokens == ["h", "i"]
    assert record.per_token_logps == [-0.1, -0.2]
    assert record.finish_reason == "stop"


def test_read_trace_returns_empty_list_when_missing(tmp_path):
    assert read_trace(tmp_path / "nonexistent.jsonl") == []


def test_proxy_forwards_captures_and_strips(tmp_path):
    upstream_port = _free_port()
    proxy_port = _free_port()
    trace = tmp_path / "trace.jsonl"

    upstream_app = _make_upstream_app(_FAKE_RESPONSE)
    upstream_server = _run_upstream(upstream_app, upstream_port)

    cfg = ProxyConfig(
        upstream_url=f"http://127.0.0.1:{upstream_port}",
        upstream_api_key="test-key",
        trace_path=str(trace),
        host="127.0.0.1",
        port=proxy_port,
        top_logprobs=5,
    )

    with InterceptionProxy(cfg) as proxy:
        assert proxy.url == f"http://127.0.0.1:{proxy_port}/v1"
        # Sanity: healthz
        r = httpx.get(f"http://127.0.0.1:{proxy_port}/healthz")
        assert r.status_code == 200
        # Chat completion round trip
        req_body = {
            "model": "openai_compatible/foo",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
        }
        r = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json=req_body,
            headers={"Authorization": "Bearer whatever"},
            timeout=10,
        )
        assert r.status_code == 200
        returned = r.json()
        # logprobs stripped from what opencode sees
        assert "logprobs" not in returned["choices"][0]
        assert returned["choices"][0]["message"]["content"] == "hi"

    # Upstream got logprobs=true injected
    forwarded = upstream_app.state.received
    assert len(forwarded) == 1
    assert forwarded[0]["logprobs"] is True
    assert forwarded[0]["top_logprobs"] == 5
    # Authorization carries upstream_api_key

    # Trace file has one line with captured logprobs
    records = read_trace(trace)
    assert len(records) == 1
    rec = records[0]
    assert rec["turn"] == 1
    assert rec["completion_tokens"] == ["h", "i"]
    assert rec["per_token_logps"] == [-0.1, -0.2]
    assert rec["finish_reason"] == "stop"
    assert rec["request"]["messages"][0]["content"] == "hi"

    upstream_server.should_exit = True


def test_proxy_handles_invalid_json_body(tmp_path):
    upstream_port = _free_port()
    proxy_port = _free_port()
    upstream_server = _run_upstream(_make_upstream_app(_FAKE_RESPONSE), upstream_port)

    cfg = ProxyConfig(
        upstream_url=f"http://127.0.0.1:{upstream_port}",
        trace_path=str(tmp_path / "trace.jsonl"),
        host="127.0.0.1",
        port=proxy_port,
    )
    with InterceptionProxy(cfg):
        r = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert r.status_code == 400

    upstream_server.should_exit = True
