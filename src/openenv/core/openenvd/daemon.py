# SPDX-License-Identifier: BSD-3-Clause

"""HTTP control surface for openenvd.

Endpoints:
    GET    /health            unauthenticated liveness probe
    POST   /tasks             register (and by default start) a sidecar task
    GET    /tasks             list task statuses
    GET    /tasks/{name}      one task's status
    POST   /tasks/{name}/start
    POST   /tasks/{name}/stop
    DELETE /tasks/{name}      stop and remove a task
    GET    /events            recent supervisor lifecycle events

The bearer token grants full administrative access, including command execution.
Only trusted operators may receive it; workloads must never have access to it.
"""

from __future__ import annotations

import argparse
import hmac
import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openenv.core.openenvd.models import TaskSpec, TaskStatus
from openenv.core.openenvd.supervisor import Supervisor


class CreateTaskRequest(TaskSpec):
    autostart: bool = True


def create_app(
    supervisor: Optional[Supervisor] = None,
    auth_token: Optional[str] = None,
) -> FastAPI:
    """Create an authenticated control API and stop its tasks on shutdown.

    The token comes from ``auth_token`` or ``OPENENVD_TOKEN``. Missing or invalid
    configuration prevents startup; the API has no unauthenticated admin mode.
    """
    token = os.environ.get("OPENENVD_TOKEN") if auth_token is None else auth_token
    if not token or not token.isascii() or not token.isprintable() or " " in token:
        raise ValueError("OPENENVD_TOKEN must be a nonempty ASCII bearer token")
    expected_token = token.encode("ascii")
    supervisor = supervisor or Supervisor()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await supervisor.shutdown()

    app = FastAPI(title="openenvd", version="0.1.0", lifespan=lifespan)
    app.state.supervisor = supervisor

    @app.middleware("http")
    async def require_bearer(request: Request, call_next):
        if request.url.path != "/health":
            scheme, _, credentials = request.headers.get("authorization", "").partition(
                " "
            )
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                credentials.encode("utf-8"), expected_token
            ):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "missing or invalid bearer token"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # FastAPI's default validation response includes inputs, which may contain
        # credentials in task environment variables or command arguments.
        return JSONResponse(status_code=422, content={"detail": "invalid request"})

    @app.exception_handler(KeyError)
    async def unknown_task(request: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": "task not found"})

    @app.exception_handler(OSError)
    async def task_operation_failed(request: Request, exc: OSError):
        return JSONResponse(
            status_code=500, content={"detail": "task operation failed"}
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/tasks", status_code=201)
    async def create_task(request: CreateTaskRequest) -> TaskStatus:
        spec = TaskSpec.model_validate(request.model_dump(exclude={"autostart"}))
        try:
            return await supervisor.register(spec, autostart=request.autostart)
        except ValueError as e:
            raise HTTPException(
                status_code=409, detail="task registration conflict"
            ) from e

    @app.get("/tasks")
    async def list_tasks() -> list[TaskStatus]:
        return supervisor.status_all()

    @app.get("/tasks/{name}")
    async def get_task(name: str) -> TaskStatus:
        return supervisor.status(name)

    @app.post("/tasks/{name}/start")
    async def start_task(name: str) -> TaskStatus:
        return await supervisor.start(name)

    @app.post("/tasks/{name}/stop")
    async def stop_task(name: str) -> TaskStatus:
        return await supervisor.stop(name)

    @app.delete("/tasks/{name}")
    async def delete_task(name: str) -> dict:
        await supervisor.unregister(name)
        return {"removed": name}

    @app.get("/events")
    async def events(after: int = -1) -> list[dict]:
        return [e.model_dump() for e in supervisor.events(after=after)]

    return app


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openenvd",
        description="OpenEnv privileged environment sidecar daemon",
        epilog="Set OPENENVD_TOKEN to a secret bearer token before starting.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8100, help="bind port")
    args = parser.parse_args(argv)
    try:
        app = create_app()
    except ValueError as e:
        parser.error(str(e))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
