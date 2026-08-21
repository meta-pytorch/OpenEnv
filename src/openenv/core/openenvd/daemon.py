# SPDX-License-Identifier: BSD-3-Clause

"""HTTP control surface for openenvd.

Endpoints:
    GET    /health            liveness + task count
    POST   /tasks             register (and by default start) a sidecar task
    GET    /tasks             list task statuses
    GET    /tasks/{name}      one task's status
    POST   /tasks/{name}/start
    POST   /tasks/{name}/stop
    DELETE /tasks/{name}      stop and remove a task
    GET    /events            recent supervisor lifecycle events

This surface is the daemon-side half of RFC 1053's policy-scoped surfaces:
orchestrators, graders, and operators talk to it; the workload never does.
"""

from __future__ import annotations

import argparse
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from openenv.core.openenvd.models import TaskSpec, TaskStatus
from openenv.core.openenvd.supervisor import Supervisor


class CreateTaskRequest(TaskSpec):
    autostart: bool = True


def create_app(supervisor: Optional[Supervisor] = None) -> FastAPI:
    supervisor = supervisor or Supervisor()
    app = FastAPI(title="openenvd", version="0.1.0")
    app.supervisor = supervisor
    app.state.supervisor = supervisor

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "tasks": len(supervisor.status_all())}

    @app.post("/tasks", status_code=201)
    async def create_task(request: CreateTaskRequest) -> TaskStatus:
        try:
            return await supervisor.register(request, autostart=request.autostart)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    @app.get("/tasks")
    async def list_tasks() -> list[TaskStatus]:
        return supervisor.status_all()

    @app.get("/tasks/{name}")
    async def get_task(name: str) -> TaskStatus:
        try:
            return supervisor.status(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.post("/tasks/{name}/start")
    async def start_task(name: str) -> TaskStatus:
        try:
            return await supervisor.start(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.post("/tasks/{name}/stop")
    async def stop_task(name: str) -> TaskStatus:
        try:
            return await supervisor.stop(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.delete("/tasks/{name}")
    async def delete_task(name: str) -> dict:
        try:
            await supervisor.unregister(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {"removed": name}

    @app.get("/events")
    async def events() -> list[dict]:
        return [e.model_dump() for e in supervisor.events()]

    return app


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openenvd",
        description="OpenEnv privileged environment sidecar daemon",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8100, help="bind port")
    args = parser.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
