# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Post-rollout summary helpers.

Read the well-known logs inside a finished (or still-live) session sandbox
and print a structured, human-readable summary: proxy turns, tool calls,
errors, and a listing of the working directory the agent produced.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .harness import OpenCodeSession


@dataclass
class RolloutSummary:
    """Structured snapshot of one rollout."""

    sandbox_id: str
    proxy_turns: list[dict[str, Any]]
    opencode_events: list[dict[str, Any]]
    workdir_listing: str
    workdir_contents: dict[str, str]

    # Derived counters
    def proxy_turn_count(self) -> int:
        return len(self.proxy_turns)

    def productive_turn_count(self) -> int:
        return sum(1 for t in self.proxy_turns if t.get("completion_tokens"))

    def total_completion_tokens(self) -> int:
        return sum(len(t.get("completion_tokens", [])) for t in self.proxy_turns)

    def tool_calls(self) -> list[str]:
        """Tool call names extracted from the opencode event stream."""
        out: list[str] = []
        for ev in self.opencode_events:
            if ev.get("type") == "tool":
                name = ev.get("part", {}).get("tool")
                if name:
                    out.append(name)
        return out

    def errors(self) -> list[str]:
        out: list[str] = []
        for ev in self.opencode_events:
            if ev.get("type") == "error":
                msg = ev.get("error", {}).get("data", {}).get("message", "")
                if msg:
                    out.append(msg)
        return out


def collect_rollout_summary(
    session: "OpenCodeSession",
    *,
    workdir: str = "/home/user/workdir",
) -> RolloutSummary:
    """Pull logs and working-directory state from a session's sandbox."""

    sbx = session.sandbox
    proxy_turns: list[dict[str, Any]] = []
    opencode_events: list[dict[str, Any]] = []

    proxy_raw = _safe_read(sbx, "/home/user/logs/agent/proxy_trace.jsonl")
    for line in proxy_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            proxy_turns.append(json.loads(line))
        except Exception:
            pass

    opencode_raw = _safe_read(sbx, "/home/user/logs/agent/opencode.jsonl")
    for line in opencode_raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            opencode_events.append(json.loads(line))
        except Exception:
            pass

    # Simple directory listing; if the sandbox has already been killed this
    # will return "" and we just skip.
    listing = ""
    try:
        result = sbx.exec(f"ls -la {workdir} 2>&1", timeout=10)
        listing = result.stdout or result.stderr or ""
    except Exception:
        pass

    # Read small files the agent created (< 10 KiB each, < 8 files).
    contents: dict[str, str] = {}
    try:
        result = sbx.exec(
            f'find {workdir} -type f -size -10k 2>/dev/null | head -8',
            timeout=10,
        )
        for path in (result.stdout or "").splitlines():
            path = path.strip()
            if not path:
                continue
            try:
                contents[path] = sbx.read_text(path)[:4000]
            except Exception:
                pass
    except Exception:
        pass

    return RolloutSummary(
        sandbox_id=sbx.sandbox_id,
        proxy_turns=proxy_turns,
        opencode_events=opencode_events,
        workdir_listing=listing,
        workdir_contents=contents,
    )


def print_rollout_summary(
    summary: RolloutSummary,
    *,
    stream=sys.stdout,
    show_bodies: bool = False,
) -> None:
    """Print a concise, scannable report."""

    p = lambda s="": print(s, file=stream, flush=True)  # noqa: E731

    p("")
    p("=" * 72)
    p(f"Rollout summary  sandbox={summary.sandbox_id}")
    p("=" * 72)

    p(f"proxy turns:          {summary.proxy_turn_count()}")
    p(f"  productive (w/ tokens): {summary.productive_turn_count()}")
    p(f"  total completion tokens: {summary.total_completion_tokens()}")

    tool_calls = summary.tool_calls()
    p(f"tool calls ({len(tool_calls)}): {', '.join(tool_calls) or '(none)'}")

    errors = summary.errors()
    if errors:
        p(f"errors ({len(errors)}):")
        for e in errors[:5]:
            p(f"  - {e[:200]}")
    else:
        p("errors: none")

    # First productive turn sample
    productive = [t for t in summary.proxy_turns if t.get("completion_tokens")]
    if productive:
        first = productive[0]
        toks = first["completion_tokens"][:10]
        lps = first["per_token_logps"][:10]
        p(f"\nfirst productive turn:")
        p(f"  finish_reason: {first.get('finish_reason')}")
        p(f"  latency:       {first.get('latency_s', 0):.2f}s")
        p(f"  first 10 tokens: {toks}")
        p(
            f"  first 10 logps:  "
            f"[{', '.join(f'{x:+.3f}' for x in lps)}]"
        )

    if summary.workdir_listing:
        p(f"\nworkdir listing:")
        for line in summary.workdir_listing.strip().splitlines()[:20]:
            p(f"  {line}")

    if summary.workdir_contents:
        p(f"\nworkdir files ({len(summary.workdir_contents)}):")
        for path, content in summary.workdir_contents.items():
            preview = content[:400].rstrip()
            p(f"  --- {path} ---")
            for line in preview.splitlines():
                p(f"    {line}")

    if show_bodies and productive:
        p("\nlast turn response body (truncated):")
        body = productive[-1].get("response", {})
        p(json.dumps(body, indent=2)[:1500])

    p("=" * 72)


def _safe_read(sandbox, path: str) -> str:
    try:
        return sandbox.read_text(path)
    except Exception:
        return ""


__all__ = [
    "RolloutSummary",
    "collect_rollout_summary",
    "print_rollout_summary",
]
