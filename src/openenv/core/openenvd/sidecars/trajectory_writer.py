# SPDX-License-Identifier: BSD-3-Clause

"""Observability sidecar: openenvd event stream -> JSONL trajectory log.

Runs as a supervised task alongside the workload and appends every
supervisor lifecycle event to a JSONL file, resuming from the last seen
sequence number across daemon restarts. Each line is a self-contained
record shaped for downstream OTLP/log export:

    {"seq": 3, "ts": 1787323324.45, "task": "agent",
     "kind": "exited", "detail": "exit_code=0"}

Usage (registered like any other sidecar task):

    POST /tasks
    {"name": "trajectory-writer",
     "argv": ["python", "-m", "openenv.core.openenvd.sidecars.trajectory_writer",
              "--daemon-url", "http://127.0.0.1:8100",
              "--out", "/tmp/trajectory.jsonl"]}

or programmatically via :func:`trajectory_writer_spec`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import httpx
from openenv.core.openenvd.models import TaskSpec


def trajectory_writer_spec(
    daemon_url: str,
    out_path: str,
    token: Optional[str] = None,
    poll_interval_s: float = 0.5,
) -> TaskSpec:
    """Build a TaskSpec registering the writer against a running daemon."""
    argv = [
        "python",
        "-m",
        "openenv.core.openenvd.sidecars.trajectory_writer",
        "--daemon-url",
        daemon_url,
        "--out",
        out_path,
        "--poll-interval",
        str(poll_interval_s),
    ]
    if token:
        argv += ["--token", token]
    return TaskSpec(
        name="trajectory-writer",
        argv=argv,
        restart_policy="on_failure",
        max_retries=5,
    )


def fetch_events(
    daemon_url: str, after: int, token: Optional[str] = None
) -> list[dict]:
    headers = {"authorization": f"Bearer {token}"} if token else {}
    resp = httpx.get(
        f"{daemon_url.rstrip('/')}/events",
        params={"after": after},
        headers=headers,
        timeout=5.0,
    )
    resp.raise_for_status()
    return resp.json()


def write_events(out_path: str, events: list[dict]) -> None:
    with open(out_path, "a") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def run(
    daemon_url: str,
    out_path: str,
    token: Optional[str] = None,
    poll_interval_s: float = 0.5,
) -> None:
    after = _read_cursor(out_path)
    while True:
        try:
            events = fetch_events(daemon_url, after, token)
            if events:
                write_events(out_path, events)
                after = events[-1]["seq"]
                _write_cursor(out_path, after)
        except Exception:
            pass
        time.sleep(poll_interval_s)


def _cursor_path(out_path: str) -> str:
    return out_path + ".cursor"


def _read_cursor(out_path: str) -> int:
    try:
        with open(_cursor_path(out_path)) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return -1


def _write_cursor(out_path: str, seq: int) -> None:
    tmp = _cursor_path(out_path) + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(seq))
    os.replace(tmp, _cursor_path(out_path))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="trajectory-writer",
        description="openenvd observability sidecar: events -> JSONL",
    )
    parser.add_argument("--daemon-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        run(args.daemon_url, args.out, args.token, args.poll_interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
