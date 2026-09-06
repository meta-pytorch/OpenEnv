# SPDX-License-Identifier: BSD-3-Clause

"""Append recent openenvd lifecycle events to a JSONL file.

This is a best-effort log of a bounded, in-memory event buffer. Each writer
process starts at the oldest retained event, so restarts can replay records
and slow polling can miss events. A standalone writer must restart with the
daemon because sequence numbers reset. This is not a durable audit trail or
a record of agent actions.

The writer is a trusted operator: OPENENVD_TOKEN grants the same control
access as the caller registering it. Supply it explicitly through the task
environment, never command-line arguments. Each JSONL line looks like:

    {"seq": 3, "ts": 1787323324.45, "task": "agent",
     "kind": "exited", "detail": "exit_code=0"}

Register the task returned by ``trajectory_writer_spec`` to run it under
the daemon's supervision. Fetch and file errors terminate the writer so
the supervisor can report the failure and apply its restart policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import httpx
from openenv.core.openenvd.models import TaskSpec


def trajectory_writer_spec(
    daemon_url: str,
    out_path: str,
    token: str,
    poll_interval_s: float = 0.5,
) -> TaskSpec:
    """Build a TaskSpec registering the writer against a running daemon."""
    argv = [
        sys.executable,
        "-m",
        "openenv.core.openenvd.sidecars.trajectory_writer",
        "--daemon-url",
        daemon_url,
        "--out",
        out_path,
        "--poll-interval",
        str(poll_interval_s),
    ]
    return TaskSpec(
        name="trajectory-writer",
        argv=argv,
        env={"OPENENVD_TOKEN": token},
        restart_policy="on_failure",
        max_retries=5,
    )


def fetch_events(daemon_url: str, after: int, token: str) -> list[dict]:
    resp = httpx.get(
        f"{daemon_url.rstrip('/')}/events",
        params={"after": after},
        headers={"authorization": f"Bearer {token}"},
        timeout=5.0,
    )
    resp.raise_for_status()
    return resp.json()


def write_events(out_path: str, events: list[dict]) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def run(
    daemon_url: str,
    out_path: str,
    token: str,
    poll_interval_s: float = 0.5,
) -> None:
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be finite and greater than zero")
    after = -1
    while True:
        events = fetch_events(daemon_url, after, token)
        if events:
            write_events(out_path, events)
            after = events[-1]["seq"]
        time.sleep(poll_interval_s)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="trajectory-writer",
        description="Best-effort openenvd lifecycle log; uses OPENENVD_TOKEN",
    )
    parser.add_argument("--daemon-url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    args = parser.parse_args(argv)
    token = os.environ.get("OPENENVD_TOKEN")
    if not token:
        parser.error("OPENENVD_TOKEN is required")
    try:
        run(args.daemon_url, args.out, token, args.poll_interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
