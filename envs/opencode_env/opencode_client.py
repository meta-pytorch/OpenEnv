# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin typed client for opencode's OpenAPI-exposed HTTP server.

Wraps the endpoints documented at https://opencode.ai/docs/server/:

- ``POST /session``                           — create a new session
- ``POST /session/:id/message``               — synchronous prompt+response
- ``POST /session/:id/prompt_async``          — fire-and-forget prompt
- ``GET  /session/:id/message``               — list all messages
- ``GET  /session/:id``                       — session state
- ``GET  /session/status``                    — all sessions
- ``POST /session/:id/abort``                 — cancel in-flight
- ``GET  /event``                             — SSE bus events

This is a dumb transport layer — no business logic. Used by the Phase-2b
session driver in :mod:`opencode_env.harness` and by the Phase-2b MCP
tools in ``trl-internal/environments/opencode/openenv/server/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

import httpx


@dataclass
class OpenCodeServerClient:
    """Sync + async helpers for the opencode serve HTTP API.

    ``base_url`` is the server's root (e.g. ``https://4096-<sbx>.e2b.app``
    or ``http://localhost:4096``). No trailing slash.
    """

    base_url: str
    timeout_s: float = 600.0
    password: str | None = None  # for OPENCODE_SERVER_PASSWORD basic auth

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    # ---------------------------------------------------------------- auth
    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.password:
            import base64

            token = base64.b64encode(f"opencode:{self.password}".encode()).decode()
            h["Authorization"] = f"Basic {token}"
        return h

    # ---------------------------------------------------------------- sync
    def openapi_spec(self) -> dict[str, Any]:
        r = httpx.get(f"{self.base_url}/doc", timeout=30, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def create_session(self, *, title: str | None = None, parent_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title
        if parent_id is not None:
            body["parentID"] = parent_id
        r = httpx.post(
            f"{self.base_url}/session",
            json=body,
            timeout=self.timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()

    def send_message(
        self,
        session_id: str,
        text: str,
        *,
        model: str | None = None,
        agent: str | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous prompt: blocks until the agent finishes the turn."""
        body: dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
        if model is not None:
            body["model"] = model
        if agent is not None:
            body["agent"] = agent
        if system is not None:
            body["system"] = system
        r = httpx.post(
            f"{self.base_url}/session/{session_id}/message",
            json=body,
            timeout=self.timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()

    def send_prompt_async(
        self,
        session_id: str,
        text: str,
        *,
        model: str | None = None,
        agent: str | None = None,
    ) -> None:
        """Fire-and-forget prompt — returns immediately (204 No Content)."""
        body: dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
        if model is not None:
            body["model"] = model
        if agent is not None:
            body["agent"] = agent
        r = httpx.post(
            f"{self.base_url}/session/{session_id}/prompt_async",
            json=body,
            timeout=self.timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()

    def list_messages(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        params = {"limit": limit} if limit is not None else None
        r = httpx.get(
            f"{self.base_url}/session/{session_id}/message",
            params=params,
            timeout=self.timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()

    def get_session(self, session_id: str) -> dict[str, Any]:
        r = httpx.get(
            f"{self.base_url}/session/{session_id}",
            timeout=self.timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()

    def get_all_status(self) -> dict[str, Any]:
        r = httpx.get(
            f"{self.base_url}/session/status",
            timeout=self.timeout_s,
            headers=self._headers(),
        )
        r.raise_for_status()
        return r.json()

    def abort(self, session_id: str) -> bool:
        r = httpx.post(
            f"{self.base_url}/session/{session_id}/abort",
            timeout=30,
            headers=self._headers(),
        )
        r.raise_for_status()
        body = r.json()
        return bool(body) if isinstance(body, bool) else bool(body.get("success"))

    # ---------------------------------------------------------- sync SSE
    def stream_events(self) -> Iterator[dict[str, Any]]:
        """Yield one decoded event dict per SSE data frame. Blocks until the
        server closes the connection or the iterator is broken out of.

        The first event is always ``{"type": "server.connected", ...}``.
        """
        with httpx.stream(
            "GET",
            f"{self.base_url}/event",
            timeout=None,
            headers=self._headers(),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data:
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    # ---------------------------------------------------------- async SSE
    async def astream_events(self) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(headers=self._headers(), timeout=None) as client:
            async with client.stream("GET", f"{self.base_url}/event") as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data:
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    # ---------------------------------------------------------- helpers
    def wait_for_ready(self, *, attempts: int = 40, interval_s: float = 0.5) -> bool:
        """Poll ``/doc`` until it returns 200, then return True.

        Returns False if it never becomes ready within ``attempts * interval_s``.
        """
        import time

        for _ in range(attempts):
            try:
                r = httpx.get(f"{self.base_url}/doc", timeout=3, headers=self._headers())
                if r.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval_s)
        return False


# A couple of helper extractors for event frames — keeps downstream code terse.

def event_session_id(event: dict[str, Any]) -> str | None:
    """Best-effort extraction of the session id from an SSE event envelope."""
    props = event.get("properties") or event.get("data") or {}
    info = props.get("info") or {}
    return info.get("sessionID") or info.get("session_id") or info.get("session") or info.get("id")


def event_part(event: dict[str, Any]) -> dict[str, Any]:
    """Return the ``part`` sub-object from a ``message.part.updated`` event,
    or an empty dict if absent."""
    props = event.get("properties") or event.get("data") or {}
    return props.get("part") or {}
