"""Hugging Face-backed provider for OpenEnv environment servers."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import suppress
from typing import Any, Dict, Optional

import httpx
import requests
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
from websockets.asyncio.client import connect as ws_connect

from .providers import ContainerProvider


_DEFAULT_PORT = 8000
_DEFAULT_COMMAND = "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _job_port_url(job: Any, port: int) -> str | None:
    for url in job.status.expose_urls or []:
        if f"--{port}." in url:
            return str(url)
    return None


def _to_ws_url(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


class _LocalAuthProxy:
    def __init__(self, *, target_url: str, token: str):
        self.target_url = target_url.rstrip("/")
        self.token = token
        self.port = _find_available_port()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> str:
        app = FastAPI()

        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def proxy_http(path: str, request: Request) -> Response:
            query = request.url.query
            target = f"{self.target_url}/{path}"
            if query:
                target = f"{target}?{query}"
            headers = {
                key: value
                for key, value in request.headers.items()
                if key.lower() not in _HOP_BY_HOP_HEADERS
            }
            headers["authorization"] = f"Bearer {self.token}"
            async with httpx.AsyncClient(follow_redirects=True) as client:
                upstream = await client.request(
                    request.method,
                    target,
                    content=await request.body(),
                    headers=headers,
                    timeout=60.0,
                )
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in _HOP_BY_HOP_HEADERS
            }
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=response_headers,
            )

        @app.websocket("/{path:path}")
        async def proxy_websocket(path: str, websocket: WebSocket) -> None:
            query = websocket.url.query
            target = f"{_to_ws_url(self.target_url)}/{path}"
            if query:
                target = f"{target}?{query}"
            await websocket.accept()
            async with ws_connect(
                target,
                additional_headers={"Authorization": f"Bearer {self.token}"},
            ) as upstream:
                to_upstream = asyncio.create_task(
                    self._client_to_upstream(websocket, upstream)
                )
                to_client = asyncio.create_task(
                    self._upstream_to_client(websocket, upstream)
                )
                done, pending = await asyncio.wait(
                    {to_upstream, to_client},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    with suppress(ConnectionClosed, WebSocketDisconnect):
                        task.result()

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("HF sandbox auth proxy failed to start")
            time.sleep(0.05)
        return self.base_url

    async def _client_to_upstream(self, websocket: WebSocket, upstream: Any) -> None:
        async for message in websocket.iter_text():
            await upstream.send(message)

    async def _upstream_to_client(self, websocket: WebSocket, upstream: Any) -> None:
        async for message in upstream:
            if isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                await websocket.send_text(message)

    def stop(self) -> None:
        if self._server is None or self._thread is None:
            return
        self._server.should_exit = True
        self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None


class HFSandboxProvider(ContainerProvider):
    """Run an OpenEnv server on Hugging Face infrastructure."""

    def __init__(
        self,
        *,
        flavor: str = "cpu-basic",
        namespace: str | None = None,
        token: str | None = None,
        command: str = _DEFAULT_COMMAND,
        timeout: int | float | str | None = "24h",
        startup_timeout_s: float = 120.0,
    ):
        self.flavor = flavor
        self.namespace = namespace
        self.token = token
        self.command = command
        self.timeout = timeout
        self.startup_timeout_s = startup_timeout_s
        self._api = HfApi(token=token)
        self._job: Any = None
        self._job_namespace: str | None = None
        self._proxy: _LocalAuthProxy | None = None

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported HFSandboxProvider options: {unknown}")
        if self._job is not None:
            raise RuntimeError("HFSandboxProvider already has an active job")

        bind_port = port or _DEFAULT_PORT
        if bind_port != _DEFAULT_PORT:
            raise ValueError(
                f"HFSandboxProvider only supports port {_DEFAULT_PORT} "
                f"(got {bind_port})."
            )

        effective_token = self.token or get_token()
        if not effective_token:
            raise ValueError(
                "HFSandboxProvider requires a Hugging Face token. "
                "Pass token= or run `hf auth login`."
            )

        self._job = self._api.run_job(
            image=image,
            command=["sh", "-lc", self.command],
            env=env_vars,
            flavor=self.flavor,
            timeout=self.timeout,
            labels={"openenv-provider": "hf-sandbox"},
            expose=[bind_port],
            namespace=self.namespace,
            token=effective_token,
        )
        self._job_namespace = self.namespace or self._job.owner.name
        target_url = self._wait_for_job_url(bind_port)
        self._proxy = _LocalAuthProxy(target_url=target_url, token=effective_token)
        return self._proxy.start()

    def _wait_for_job_url(self, port: int) -> str:
        deadline = time.time() + self.startup_timeout_s
        target_url = _job_port_url(self._job, port)
        while target_url is None and time.time() < deadline:
            time.sleep(0.5)
            self._job = self._api.inspect_job(
                job_id=self._job.id,
                namespace=self._job_namespace,
                token=self.token,
            )
            target_url = _job_port_url(self._job, port)
        if target_url is None:
            raise RuntimeError(
                f"HF job did not expose port {port} within "
                f"{self.startup_timeout_s:.1f}s"
            )
        return target_url

    def stop_container(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        if self._job is not None:
            self._api.cancel_job(
                job_id=self._job.id,
                namespace=self._job_namespace,
                token=self.token,
            )
            self._job = None
            self._job_namespace = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        deadline = time.time() + timeout_s
        health_url = f"{base_url}/health"
        while time.time() < deadline:
            response = requests.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"HF sandbox job at {base_url} did not become ready within {timeout_s}s"
        )

    def close(self) -> None:
        self.stop_container()


__all__ = ["HFSandboxProvider"]
