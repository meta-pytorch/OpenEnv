# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenvd daemon HTTP control surface."""

import httpx
import pytest
from openenv.core.openenvd import daemon
from openenv.core.openenvd.daemon import create_app
from openenv.core.openenvd.models import TaskState

TOKEN = "test-daemon-token"


@pytest.fixture
async def client():
    app = create_app(auth_token=TOKEN)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://openenvd",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as c:
            yield c, app


class TestHealth:
    async def test_health(self, client):
        c, _ = client
        c.headers.pop("authorization")
        resp = await c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAuthentication:
    @pytest.mark.parametrize(
        "token", [None, "", " ", "bad token", "bad\n", "\x00", "é"]
    )
    def test_missing_or_invalid_token_prevents_app_creation(self, monkeypatch, token):
        monkeypatch.delenv("OPENENVD_TOKEN", raising=False)
        with pytest.raises(ValueError, match="OPENENVD_TOKEN"):
            create_app(auth_token=token)

    async def test_token_can_come_from_environment(self, monkeypatch):
        monkeypatch.setenv("OPENENVD_TOKEN", TOKEN)
        app = create_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://openenvd"
        ) as c:
            assert (await c.get("/tasks")).status_code == 401
            response = await c.get(
                "/tasks", headers={"Authorization": f"Bearer {TOKEN}"}
            )
            assert response.status_code == 200

    @pytest.mark.parametrize(
        "authorization", [None, b"Bearer wrong", b"Basic wrong", b"Bearer \xff"]
    )
    async def test_invalid_auth_cannot_register_or_spawn(self, client, authorization):
        c, app = client
        c.headers.pop("authorization")
        headers = {} if authorization is None else {b"authorization": authorization}
        response = await c.post(
            "/tasks",
            headers=headers,
            json={"name": "unauthorized", "argv": ["sleep", "30"], "auto_uid": False},
        )
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert app.state.supervisor.status_all() == []
        assert app.state.supervisor.events() == []

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/tasks"),
            ("GET", "/tasks/secret"),
            ("POST", "/tasks/secret/start"),
            ("POST", "/tasks/secret/stop"),
            ("DELETE", "/tasks/secret"),
            ("GET", "/events"),
            ("GET", "/docs"),
            ("GET", "/openapi.json"),
        ],
    )
    async def test_all_control_endpoints_require_auth(self, client, method, path):
        c, _ = client
        c.headers.pop("authorization")
        assert (await c.request(method, path)).status_code == 401


class TestTaskEndpoints:
    async def test_register_and_start_task(self, client):
        c, _ = client
        resp = await c.post(
            "/tasks",
            json={"name": "sleeper", "argv": ["sleep", "30"], "auto_uid": False},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "sleeper"
        assert body["state"] == TaskState.RUNNING.value
        assert (await c.get("/tasks/sleeper")).json() == body

    async def test_register_without_autostart(self, client):
        c, _ = client
        resp = await c.post(
            "/tasks",
            json={
                "name": "idle",
                "argv": ["sleep", "30"],
                "autostart": False,
                "auto_uid": False,
            },
        )
        assert resp.status_code == 201
        body = (await c.get("/tasks/idle")).json()
        assert body["state"] == TaskState.REGISTERED.value

    async def test_duplicate_registration_conflict(self, client):
        c, _ = client
        payload = {
            "name": "dup",
            "argv": ["sleep", "30"],
            "autostart": False,
            "auto_uid": False,
        }
        assert (await c.post("/tasks", json=payload)).status_code == 201
        resp = await c.post("/tasks", json=payload)
        assert resp.status_code == 409
        assert resp.json() == {"detail": "task registration conflict"}

    async def test_invalid_spec_rejected(self, client):
        c, _ = client
        resp = await c.post("/tasks", json={"name": "bad", "argv": []})
        assert resp.status_code == 422

    async def test_validation_errors_do_not_echo_secrets(self, client):
        c, _ = client
        response = await c.post(
            "/tasks",
            json={
                "name": "bad",
                "argv": ["true"],
                "env": {"TOKEN": ["private-credential"]},
                "auto_uid": False,
            },
        )
        assert response.status_code == 422
        assert response.json() == {"detail": "invalid request"}

    async def test_list_tasks(self, client):
        c, _ = client
        for name in ("a", "b"):
            await c.post(
                "/tasks",
                json={
                    "name": name,
                    "argv": ["true"],
                    "autostart": False,
                    "auto_uid": False,
                },
            )
        resp = await c.get("/tasks")
        assert resp.status_code == 200
        names = {t["name"] for t in resp.json()}
        assert names == {"a", "b"}

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/tasks/nope"),
            ("POST", "/tasks/nope/start"),
            ("POST", "/tasks/nope/stop"),
            ("DELETE", "/tasks/nope"),
        ],
    )
    async def test_unknown_task_404(self, client, method, path):
        c, _ = client
        response = await c.request(method, path)
        assert response.status_code == 404
        assert response.json() == {"detail": "task not found"}

    @pytest.mark.parametrize("autostart", [True, False])
    async def test_spawn_failures_are_redacted_and_leave_failed_status(
        self, client, tmp_path, autostart
    ):
        c, _ = client
        response = await c.post(
            "/tasks",
            json={
                "name": "missing",
                "argv": [str(tmp_path / "private-credential")],
                "autostart": autostart,
                "auto_uid": False,
            },
        )
        if not autostart:
            assert response.status_code == 201
            response = await c.post("/tasks/missing/start")
        assert response.status_code == 500
        assert response.json() == {"detail": "task operation failed"}
        status = (await c.get("/tasks/missing")).json()
        assert status["state"] == TaskState.FAILED.value
        assert status["pid"] is None

    async def test_stop_endpoint(self, client):
        c, _ = client
        await c.post(
            "/tasks", json={"name": "s", "argv": ["sleep", "30"], "auto_uid": False}
        )
        resp = await c.post("/tasks/s/stop")
        assert resp.status_code == 200
        assert resp.json()["state"] == TaskState.STOPPED.value

    async def test_start_after_stop(self, client):
        c, _ = client
        await c.post(
            "/tasks", json={"name": "r", "argv": ["sleep", "30"], "auto_uid": False}
        )
        await c.post("/tasks/r/stop")
        resp = await c.post("/tasks/r/start")
        assert resp.status_code == 200
        assert resp.json()["state"] == TaskState.RUNNING.value

    async def test_delete_task(self, client):
        c, _ = client
        await c.post(
            "/tasks", json={"name": "d", "argv": ["sleep", "30"], "auto_uid": False}
        )
        resp = await c.delete("/tasks/d")
        assert resp.status_code == 200
        assert (await c.get("/tasks/d")).status_code == 404

    async def test_events_endpoint(self, client):
        c, _ = client
        await c.post("/tasks", json={"name": "e", "argv": ["true"], "auto_uid": False})
        resp = await c.get("/events")
        assert resp.status_code == 200
        kinds = [e["kind"] for e in resp.json()]
        assert "started" in kinds


async def test_lifespan_stops_running_tasks():
    app = create_app(auth_token=TOKEN)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://openenvd",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as c:
            response = await c.post(
                "/tasks",
                json={"name": "sleeper", "argv": ["sleep", "30"], "auto_uid": False},
            )
            assert response.status_code == 201
            assert response.json()["state"] == TaskState.RUNNING.value
    status = app.state.supervisor.status("sleeper")
    assert status.state == TaskState.STOPPED
    assert status.pid is None


class TestCommandLine:
    def test_missing_token_prevents_server_start(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENENVD_TOKEN", raising=False)

        def unexpected_start(*args, **kwargs):
            pytest.fail("server started without authentication")

        monkeypatch.setattr(daemon.uvicorn, "run", unexpected_start)
        with pytest.raises(SystemExit) as error:
            daemon.main([])
        assert error.value.code == 2
        assert "OPENENVD_TOKEN" in capsys.readouterr().err

    def test_default_bind_is_loopback(self, monkeypatch):
        monkeypatch.setenv("OPENENVD_TOKEN", TOKEN)
        calls = []
        monkeypatch.setattr(
            daemon.uvicorn, "run", lambda app, **kwargs: calls.append(kwargs)
        )
        daemon.main([])
        assert calls == [{"host": "127.0.0.1", "port": 8100}]

    def test_help_uses_environment_token_not_command_argument(self, capsys):
        with pytest.raises(SystemExit) as error:
            daemon.main(["--help"])
        assert error.value.code == 0
        help_text = capsys.readouterr().out
        assert "OPENENVD_TOKEN" in help_text
        assert "--auth-token" not in help_text
