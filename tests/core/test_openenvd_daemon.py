# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenvd daemon HTTP control surface."""

import httpx
import pytest
from openenv.core.openenvd.daemon import create_app
from openenv.core.openenvd.models import TaskState


@pytest.fixture
async def client():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://openenvd") as c:
        yield c, app
    await app.supervisor.shutdown()


class TestHealth:
    async def test_health(self, client):
        c, _ = client
        resp = await c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["tasks"] == 0


class TestTaskEndpoints:
    async def test_register_and_start_task(self, client):
        c, _ = client
        resp = await c.post(
            "/tasks",
            json={"name": "sleeper", "argv": ["sleep", "30"]},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "sleeper"
        assert body["state"] in (TaskState.STARTING.value, TaskState.RUNNING.value)

        for _ in range(100):
            body = (await c.get("/tasks/sleeper")).json()
            if body["state"] == TaskState.RUNNING.value:
                break
        assert body["state"] == TaskState.RUNNING.value

    async def test_register_without_autostart(self, client):
        c, _ = client
        resp = await c.post(
            "/tasks",
            json={"name": "idle", "argv": ["sleep", "30"], "autostart": False},
        )
        assert resp.status_code == 201
        body = (await c.get("/tasks/idle")).json()
        assert body["state"] == TaskState.REGISTERED.value

    async def test_duplicate_registration_conflict(self, client):
        c, _ = client
        payload = {"name": "dup", "argv": ["sleep", "30"], "autostart": False}
        assert (await c.post("/tasks", json=payload)).status_code == 201
        resp = await c.post("/tasks", json=payload)
        assert resp.status_code == 409

    async def test_invalid_spec_rejected(self, client):
        c, _ = client
        resp = await c.post("/tasks", json={"name": "bad", "argv": []})
        assert resp.status_code == 422

    async def test_list_tasks(self, client):
        c, _ = client
        await c.post("/tasks", json={"name": "a", "argv": ["true"], "autostart": False})
        await c.post("/tasks", json={"name": "b", "argv": ["true"], "autostart": False})
        resp = await c.get("/tasks")
        assert resp.status_code == 200
        names = {t["name"] for t in resp.json()}
        assert names == {"a", "b"}

    async def test_get_unknown_task_404(self, client):
        c, _ = client
        assert (await c.get("/tasks/nope")).status_code == 404

    async def test_stop_endpoint(self, client):
        c, _ = client
        await c.post("/tasks", json={"name": "s", "argv": ["sleep", "30"]})
        resp = await c.post("/tasks/s/stop")
        assert resp.status_code == 200
        assert resp.json()["state"] == TaskState.STOPPED.value

    async def test_start_after_stop(self, client):
        c, _ = client
        await c.post("/tasks", json={"name": "r", "argv": ["sleep", "30"]})
        await c.post("/tasks/r/stop")
        resp = await c.post("/tasks/r/start")
        assert resp.status_code == 200
        assert resp.json()["state"] in (
            TaskState.STARTING.value,
            TaskState.RUNNING.value,
        )

    async def test_delete_task(self, client):
        c, _ = client
        await c.post("/tasks", json={"name": "d", "argv": ["sleep", "30"]})
        resp = await c.delete("/tasks/d")
        assert resp.status_code == 200
        assert (await c.get("/tasks/d")).status_code == 404

    async def test_events_endpoint(self, client):
        c, _ = client
        await c.post("/tasks", json={"name": "e", "argv": ["true"]})
        resp = await c.get("/events")
        assert resp.status_code == 200
        kinds = [e["kind"] for e in resp.json()]
        assert "started" in kinds
