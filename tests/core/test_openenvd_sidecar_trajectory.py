# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for the trajectory-writer sidecar.

These run a real openenvd daemon on a real socket and register the writer
through the public /tasks API, exactly as an external operator would.
"""

import asyncio
import json
import threading
import time

import httpx
import uvicorn
from openenv.core.openenvd.daemon import create_app
from openenv.core.openenvd.sidecars.trajectory_writer import (
    fetch_events,
    trajectory_writer_spec,
)


class _Daemon:
    def __init__(self, token=None):
        self.app = create_app(auth_token=token)
        config = uvicorn.Config(self.app, host="127.0.0.1", port=0, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        assert self.server.started
        self.url = (
            f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"
        )
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        return False


async def _wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return False


class TestTrajectoryWriterSpec:
    def test_spec_is_valid_task(self):
        spec = trajectory_writer_spec("http://127.0.0.1:8100", "/tmp/t.jsonl")
        assert spec.name == "trajectory-writer"
        assert spec.restart_policy == "on_failure"
        assert "trajectory_writer" in " ".join(spec.argv)
        assert "--daemon-url" in spec.argv

    def test_fetch_events_filters_by_seq(self):
        with _Daemon() as daemon:
            assert fetch_events(daemon.url, after=-1) == []
            daemon.app.supervisor._record("t", "started")
            daemon.app.supervisor._record("t", "exited")
            all_events = fetch_events(daemon.url, after=-1)
            assert [e["seq"] for e in all_events] == [0, 1]
            filtered = fetch_events(daemon.url, after=all_events[0]["seq"])
            assert [e["seq"] for e in filtered] == [1]


class TestSidecarStacking:
    async def test_registered_via_api_writes_trajectory(self, tmp_path):
        out_path = tmp_path / "trajectory.jsonl"
        with _Daemon() as daemon:
            spec = trajectory_writer_spec(daemon.url, str(out_path))
            payload = spec.model_dump()
            payload["argv"] = [
                "python",
                "-m",
                "openenv.core.openenvd.sidecars.trajectory_writer",
                "--daemon-url",
                daemon.url,
                "--out",
                str(out_path),
                "--poll-interval",
                "0.2",
            ]
            async with httpx.AsyncClient(base_url=daemon.url) as c:
                resp = await c.post("/tasks", json=payload)
                assert resp.status_code == 201

                await c.post(
                    "/tasks",
                    json={"name": "worker", "argv": ["sh", "-c", "exit 0"]},
                )
                assert await _wait_for(out_path.exists)

                lines = out_path.read_text().splitlines()
                events = [json.loads(line) for line in lines]
                kinds = {e["kind"] for e in events}
                assert "registered" in kinds
                assert "started" in kinds

    async def test_writer_resume_has_no_gaps_or_duplicates(self, tmp_path):
        out_path = tmp_path / "trajectory.jsonl"
        with _Daemon() as daemon:
            argv = [
                "python",
                "-m",
                "openenv.core.openenvd.sidecars.trajectory_writer",
                "--daemon-url",
                daemon.url,
                "--out",
                str(out_path),
                "--poll-interval",
                "0.2",
            ]
            proc = await asyncio.create_subprocess_exec(*argv)
            try:
                daemon.app.supervisor._record("before", "started")
                assert await _wait_for(
                    lambda: out_path.exists() and "before" in out_path.read_text()
                )
            finally:
                proc.terminate()
                await proc.wait()

            daemon.app.supervisor._record("mid", "exited")

            proc = await asyncio.create_subprocess_exec(*argv)
            try:
                assert await _wait_for(
                    lambda: "mid" in (out_path.read_text() if out_path.exists() else "")
                )
            finally:
                proc.terminate()
                await proc.wait()

            events = [json.loads(line) for line in out_path.read_text().splitlines()]
            tasks = [e["task"] for e in events]
            assert tasks.count("before") == 1, f"duplicated: {tasks}"
            assert tasks.count("mid") == 1, f"gap: {tasks}"
            seqs = [e["seq"] for e in events]
            assert seqs == sorted(seqs), f"out of order: {seqs}"

    async def test_writer_with_auth_token(self, tmp_path):
        out_path = tmp_path / "trajectory.jsonl"
        with _Daemon(token="secret") as daemon:
            argv = [
                "python",
                "-m",
                "openenv.core.openenvd.sidecars.trajectory_writer",
                "--daemon-url",
                daemon.url,
                "--out",
                str(out_path),
                "--token",
                "secret",
                "--poll-interval",
                "0.2",
            ]
            proc = await asyncio.create_subprocess_exec(*argv)
            try:
                daemon.app.supervisor._record("authed", "started")
                assert await _wait_for(
                    lambda: out_path.exists() and "authed" in out_path.read_text()
                )
            finally:
                proc.terminate()
                await proc.wait()

    async def test_writer_rejected_without_token(self, tmp_path):
        out_path = tmp_path / "trajectory.jsonl"
        with _Daemon(token="secret") as daemon:
            argv = [
                "python",
                "-m",
                "openenv.core.openenvd.sidecars.trajectory_writer",
                "--daemon-url",
                daemon.url,
                "--out",
                str(out_path),
                "--poll-interval",
                "0.2",
            ]
            proc = await asyncio.create_subprocess_exec(*argv)
            try:
                daemon.app.supervisor._record("denied", "started")
                await asyncio.sleep(1.5)
                assert not out_path.exists(), "writer wrote events without a token"
            finally:
                proc.terminate()
                await proc.wait()
