# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the lifecycle logger through an authenticated daemon."""

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from openenv.core.openenvd.daemon import create_app
from openenv.core.openenvd.sidecars import trajectory_writer


class _Daemon:
    token = "test-operator-token"

    def __init__(self):
        self.app = create_app(auth_token=self.token)
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
        self.headers = {"authorization": f"Bearer {self.token}"}
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()

    def register(self, name):
        response = httpx.post(
            f"{self.url}/tasks",
            headers=self.headers,
            json={
                "name": name,
                "argv": ["unused"],
                "auto_uid": False,
                "autostart": False,
            },
        )
        assert response.status_code == 201


async def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


def _writer_spec(daemon, out_path):
    spec = trajectory_writer.trajectory_writer_spec(
        daemon.url, str(out_path), token=daemon.token, poll_interval_s=0.1
    )
    # Source checkout execution and same-user mode are explicit test choices.
    spec.env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    spec.env["PATH"] = os.defpath
    spec.auto_uid = False
    return spec


@asynccontextmanager
async def _writer(spec):
    proc = await asyncio.create_subprocess_exec(*spec.argv, env=spec.env)
    try:
        yield proc
    finally:
        if proc.returncode is None:
            proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)


def test_token_is_passed_only_through_explicit_environment():
    spec = trajectory_writer.trajectory_writer_spec(
        "http://127.0.0.1:8100", "/tmp/events.jsonl", token="private-token"
    )
    assert spec.env == {"OPENENVD_TOKEN": "private-token"}
    assert "private-token" not in " ".join(spec.argv)
    assert "--token" not in spec.argv
    assert spec.restart_policy == "on_failure"


def test_fetch_events_filters_by_seq():
    with _Daemon() as daemon:
        assert trajectory_writer.fetch_events(daemon.url, -1, daemon.token) == []
        daemon.register("first")
        daemon.register("second")
        events = trajectory_writer.fetch_events(daemon.url, -1, daemon.token)
        assert [event["task"] for event in events] == ["first", "second"]
        filtered = trajectory_writer.fetch_events(
            daemon.url, events[0]["seq"], daemon.token
        )
        assert filtered == events[1:]


async def test_registered_via_api_writes_events_without_exposing_token(tmp_path):
    out_path = tmp_path / "events.jsonl"
    with _Daemon() as daemon:
        spec = _writer_spec(daemon, out_path)
        async with httpx.AsyncClient(
            base_url=daemon.url, headers=daemon.headers
        ) as client:
            response = await client.post("/tasks", json=spec.model_dump())
            assert response.status_code == 201
            daemon.register("worker")
            assert await _wait_for(
                lambda: out_path.exists() and "worker" in out_path.read_text()
            )
            assert (
                await client.post("/tasks/trajectory-writer/stop")
            ).status_code == 200
            status = await client.get("/tasks/trajectory-writer")
            assert daemon.token not in status.text

    text = out_path.read_text()
    events = [json.loads(line) for line in text.splitlines()]
    assert {event["kind"] for event in events} >= {"registered", "started"}
    assert len({event["seq"] for event in events}) == len(events)
    assert daemon.token not in text


async def test_restart_replays_retained_events_without_a_cursor_file(tmp_path):
    out_path = tmp_path / "events.jsonl"
    with _Daemon() as daemon:
        daemon.register("before")
        spec = _writer_spec(daemon, out_path)
        async with _writer(spec):
            assert await _wait_for(
                lambda: out_path.exists() and "before" in out_path.read_text()
            )
        daemon.register("between")
        async with _writer(spec):
            assert await _wait_for(lambda: "between" in out_path.read_text())

    events = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert [event["task"] for event in events] == ["before", "before", "between"]
    assert not Path(str(out_path) + ".cursor").exists()


async def test_invalid_token_exits_with_error(tmp_path):
    out_path = tmp_path / "events.jsonl"
    with _Daemon() as daemon:
        spec = _writer_spec(daemon, out_path)
        spec.env["OPENENVD_TOKEN"] = "incorrect-token"
        proc = await asyncio.create_subprocess_exec(
            *spec.argv, env=spec.env, stderr=asyncio.subprocess.PIPE
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            assert proc.returncode != 0
            assert b"401" in stderr
            assert b"incorrect-token" not in stderr
            assert not out_path.exists()
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()


def test_missing_token_fails_before_polling(monkeypatch, capsys):
    monkeypatch.delenv("OPENENVD_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        trajectory_writer.main(
            ["--daemon-url", "http://127.0.0.1:8100", "--out", "events.jsonl"]
        )
    assert exc.value.code == 2
    assert "OPENENVD_TOKEN is required" in capsys.readouterr().err


def test_output_failure_is_not_silenced(monkeypatch, tmp_path):
    monkeypatch.setattr(trajectory_writer, "fetch_events", lambda *args: [{"seq": 0}])
    with pytest.raises(IsADirectoryError):
        trajectory_writer.run("http://127.0.0.1:8100", str(tmp_path), "token")


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_poll_interval_must_be_positive_and_finite(interval, tmp_path):
    with pytest.raises(ValueError, match="finite and greater than zero"):
        trajectory_writer.run(
            "http://127.0.0.1:8100", str(tmp_path), "token", poll_interval_s=interval
        )
