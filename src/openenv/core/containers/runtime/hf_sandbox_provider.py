# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hugging Face-backed provider for OpenEnv environment servers."""

from __future__ import annotations

import asyncio
import hashlib
import socket
import threading
import time
from contextlib import suppress
from typing import Any, Literal

import requests
import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from .providers import ContainerProvider


_DEFAULT_PORT = 8000
_POOLED_SERVER_COMMAND = (
    'export SBX_PROXY_DIR="${SBX_PROXY_DIR:-$HOME/.sbx/proxy}"; '
    'mkdir -p "$SBX_PROXY_DIR"; '
    'exec server >"$HOME/openenv-server.log" 2>&1'
)
_DEDICATED_SERVER_COMMAND = (
    'unset SBX_PROXY_DIR; exec server >"$HOME/openenv-server.log" 2>&1'
)
# Compatibility alias for callers that imported the old private constant.
_SERVER_COMMAND = _POOLED_SERVER_COMMAND
_MAX_WS_MESSAGE_SIZE = 100 * 1024 * 1024
_MAX_HTTP_RESPONSE_SIZE = 100 * 1024 * 1024
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
_POOLS: dict[tuple[str, str], Any] = {}
_POOL_LOCK = threading.Lock()


class _NoRedirectWebSocketConnect(ws_connect):
    def process_redirect(self, exc: Exception) -> Exception | str:
        return exc


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def _to_ws_url(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def _pool_name(image: str, flavor: str) -> str:
    digest = hashlib.sha1(f"{image}\0{flavor}".encode("utf-8")).hexdigest()[:12]
    return f"openenv-{digest}"


def _get_sandbox_pool_cls() -> Any:
    try:
        from huggingface_hub import SandboxPool
    except ImportError as exc:
        raise RuntimeError(
            "HFSandboxProvider requires a huggingface_hub version with "
            "SandboxPool.create and Sandbox.proxy_url_for support."
        ) from exc
    return SandboxPool


def _get_sandbox_cls() -> Any:
    try:
        from huggingface_hub import Sandbox
    except ImportError as exc:
        raise RuntimeError(
            "Dedicated HFSandboxProvider execution requires huggingface_hub>=1.22.0. "
            "Install it with `pip install 'openenv[hf-sandbox]'`."
        ) from exc
    if not hasattr(Sandbox, "create"):
        raise RuntimeError(
            "Dedicated HFSandboxProvider execution requires huggingface_hub>=1.22.0. "
            "Install it with `pip install 'openenv[hf-sandbox]'`."
        )
    return Sandbox


def _get_pool(image: str, flavor: str) -> Any:
    key = (image, flavor)
    with _POOL_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            SandboxPool = _get_sandbox_pool_cls()
            pool = SandboxPool(
                image=image,
                flavor=flavor,
                name=_pool_name(image, flavor),
            )
            _POOLS[key] = pool
        return pool


class _LocalAuthProxy:
    def __init__(self, *, target_url: str, headers: dict[str, str]):
        self.target_url = target_url.rstrip("/")
        self.headers = headers
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
            headers = dict(self.headers)
            content_type = request.headers.get("content-type")
            if content_type is not None:
                headers["Content-Type"] = content_type
            body = await request.body()
            try:
                upstream = await asyncio.to_thread(
                    requests.request,
                    request.method,
                    target,
                    data=body,
                    headers=headers,
                    timeout=60.0,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException:
                return Response(
                    content=b"upstream HF job unreachable",
                    status_code=502,
                )
            try:
                if 300 <= upstream.status_code < 400:
                    return Response(
                        content=b"upstream HF job redirects are not allowed",
                        status_code=502,
                    )
                content = upstream.raw.read(
                    _MAX_HTTP_RESPONSE_SIZE + 1, decode_content=True
                )
                if len(content) > _MAX_HTTP_RESPONSE_SIZE:
                    return Response(
                        content=b"upstream HF job response exceeded proxy limit",
                        status_code=502,
                    )
                response_headers = {
                    key: value
                    for key, value in upstream.headers.items()
                    if key.lower() not in _HOP_BY_HOP_HEADERS
                }
                return Response(
                    content=content,
                    status_code=upstream.status_code,
                    headers=response_headers,
                )
            finally:
                upstream.close()

        @app.websocket("/{path:path}")
        async def proxy_websocket(path: str, websocket: WebSocket) -> None:
            query = websocket.url.query
            target = f"{_to_ws_url(self.target_url)}/{path}"
            if query:
                target = f"{target}?{query}"
            headers = {**self.headers, "accept": "*/*"}
            upstream = await self._connect_upstream_websocket(target, headers)
            await websocket.accept()
            try:
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
            finally:
                with suppress(Exception):
                    await upstream.close()

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

    async def _connect_upstream_websocket(
        self, target: str, headers: dict[str, str]
    ) -> Any:
        last_error: Exception | None = None
        for _ in range(5):
            try:
                return await _NoRedirectWebSocketConnect(
                    target,
                    additional_headers=headers,
                    max_size=_MAX_WS_MESSAGE_SIZE,
                    compression=None,
                )
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5)
        assert last_error is not None
        raise last_error

    async def _client_to_upstream(self, websocket: WebSocket, upstream: Any) -> None:
        # EnvClient sends JSON text frames; binary frames are only relayed downstream.
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
        image: str,
        env_vars: dict[str, str] | None = None,
        secrets: dict[str, str] | None = None,
        flavor: str = "cpu-basic",
        mode: Literal["pooled", "dedicated"] = "pooled",
    ):
        if mode not in {"pooled", "dedicated"}:
            raise ValueError("HFSandboxProvider mode must be 'pooled' or 'dedicated'")
        if mode == "pooled" and secrets:
            raise ValueError("Encrypted secrets require dedicated HF sandbox mode")
        self.image = image
        self._env_vars = dict(env_vars) if env_vars is not None else None
        self._secrets = dict(secrets) if secrets is not None else None
        self.flavor = flavor
        self._mode = mode
        self._sandbox: Any = None
        self._server_process: Any = None
        self._proxy: _LocalAuthProxy | None = None

    @property
    def mode(self) -> Literal["pooled", "dedicated"]:
        return self._mode

    @property
    def env_vars(self) -> dict[str, str] | None:
        return dict(self._env_vars) if self._env_vars is not None else None

    @property
    def secrets(self) -> dict[str, str] | None:
        return dict(self._secrets) if self._secrets is not None else None

    @property
    def isolation_mode(self) -> Literal["pooled", "dedicated", "unknown"]:
        if self._sandbox is None:
            return self._mode
        try:
            host_id = self._sandbox.host_id
        except (AttributeError, RuntimeError):
            return "unknown"
        return "dedicated" if host_id is None else "pooled"

    def _create_sandbox(self, *, image: str, env_vars: dict[str, str] | None) -> Any:
        if self.mode == "dedicated":
            Sandbox = _get_sandbox_cls()
            create_kwargs: dict[str, Any] = {
                "image": image,
                "flavor": self.flavor,
                "env": env_vars,
            }
            if self._secrets is not None:
                create_kwargs["secrets"] = dict(self._secrets)
            return Sandbox.create(**create_kwargs)
        pool = _get_pool(image, self.flavor)
        return pool.create(env=env_vars)

    def start_container(
        self,
        image: str | None = None,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        if self._sandbox is not None:
            raise RuntimeError("HFSandboxProvider already has an active service")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(
                f"HFSandboxProvider does not support start_container kwargs: {unsupported}"
            )

        if port not in (None, _DEFAULT_PORT):
            raise ValueError(
                f"HFSandboxProvider only supports port {_DEFAULT_PORT} (got {port})."
            )

        effective_image = self.image if image is None else image
        effective_env = self._env_vars if env_vars is None else env_vars
        self._sandbox = self._create_sandbox(
            image=effective_image, env_vars=effective_env
        )
        try:
            if not hasattr(self._sandbox, "proxy_url_for") or not hasattr(
                self._sandbox, "proxy_headers"
            ):
                raise RuntimeError(
                    "HFSandboxProvider requires a huggingface_hub version with "
                    "Sandbox.proxy_url_for and Sandbox.proxy_headers support."
                )
            self._server_process = self._sandbox.run(
                (
                    _DEDICATED_SERVER_COMMAND
                    if self.mode == "dedicated"
                    else _POOLED_SERVER_COMMAND
                ),
                shell=True,
                background=True,
            )
            self._proxy = _LocalAuthProxy(
                target_url=self._sandbox.proxy_url_for(_DEFAULT_PORT, "/"),
                headers=self._sandbox.proxy_headers,
            )
            return self._proxy.start()
        except Exception:
            with suppress(Exception):
                self.stop_container()
            raise

    def stop_container(self) -> None:
        proxy_error: Exception | None = None
        if self._proxy is not None:
            try:
                self._proxy.stop()
            except Exception as exc:
                proxy_error = exc
            finally:
                self._proxy = None
        if self._sandbox is not None:
            sandbox = self._sandbox
            sandbox.kill()
            if getattr(sandbox, "_killed", None) is not True:
                raise RuntimeError(
                    "Hugging Face did not confirm termination of the sandbox; "
                    "the provider retained its handle so cleanup can be retried."
                )
            self._sandbox = None
            self._server_process = None
        if proxy_error is not None:
            raise proxy_error

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        deadline = time.time() + timeout_s
        health_url = f"{base_url}/health"
        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0, allow_redirects=False)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1.0)
        raise TimeoutError(
            f"HF sandbox job at {base_url} did not become ready within {timeout_s}s"
        )

    def close(self) -> None:
        self.stop_container()


__all__ = ["HFSandboxProvider"]
