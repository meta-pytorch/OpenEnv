# SPDX-License-Identifier: BSD-3-Clause

"""Loopback MCP bridge exposing environment tools to a harness (RFC 005)."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, Optional

from .adapter import HarnessError


class HarnessMCPBridge:
    """
    Serve an in-process FastMCP server over loopback HTTP for harness use.

    The bridge runs the environment's FastMCP tool server as a
    streamable-HTTP ASGI app on `127.0.0.1` with an ephemeral port, inside a
    daemon thread with its own event loop. It carries only the tool surface:
    the harness subprocess can reach the environment's domain tools but never
    OpenEnv's orchestration API (`reset`/`step`/`state`), which lives on a
    different server entirely. This makes the RFC 005 security boundary
    structural rather than filter-based.

    Args:
        mcp_server (`FastMCP`):
            The environment's FastMCP server to expose.
        host (`str`, *optional*, defaults to `"127.0.0.1"`):
            Interface to bind. Keep this loopback-only.

    Examples:

    ```python
    bridge = HarnessMCPBridge(env.mcp_server)
    url = bridge.start()
    # pass url to the harness adapter's inject_tools()
    bridge.stop()
    ```
    """

    def __init__(self, mcp_server: Any, host: str = "127.0.0.1"):
        self._mcp_server = mcp_server
        self._host = host
        self._url: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[Any] = None
        self._startup_error: Optional[BaseException] = None

    @property
    def url(self) -> Optional[str]:
        """The bridge's MCP endpoint URL, or `None` when not running."""
        return self._url

    def start(self, timeout_s: float = 10.0) -> str:
        """
        Start serving and return the MCP endpoint URL.

        Idempotent: returns the existing URL if already running.

        Args:
            timeout_s (`float`, *optional*, defaults to `10.0`):
                Maximum time to wait for the server to come up.

        Returns:
            `str` URL of the MCP endpoint, e.g. `"http://127.0.0.1:54321/mcp"`.

        Raises:
            [`~openenv.core.harness.adapter.HarnessError`]:
                If the server fails to start within the timeout.
        """
        if self._thread is not None and self._thread.is_alive():
            assert self._url is not None
            return self._url

        import uvicorn

        app = self._mcp_server.http_app()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, 0))
        port = sock.getsockname()[1]

        config = uvicorn.Config(app, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        self._startup_error = None

        def _serve() -> None:
            try:
                server.run(sockets=[sock])
            except BaseException as exc:  # surfaced to start() below
                self._startup_error = exc
            finally:
                try:
                    sock.close()
                except OSError:
                    pass

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        self._thread = thread

        deadline = time.monotonic() + timeout_s
        while not server.started:
            if self._startup_error is not None or not thread.is_alive():
                self._teardown()
                raise HarnessError(f"MCP bridge failed to start: {self._startup_error}")
            if time.monotonic() > deadline:
                self.stop()
                raise HarnessError(f"MCP bridge did not start within {timeout_s}s")
            time.sleep(0.01)

        self._url = f"http://{self._host}:{port}/mcp"
        return self._url

    def stop(self, timeout_s: float = 5.0) -> None:
        """
        Stop the bridge server.

        Idempotent: safe to call when the bridge was never started.

        Args:
            timeout_s (`float`, *optional*, defaults to `5.0`):
                Maximum time to wait for the server thread to exit.
        """
        server = self._uvicorn_server
        thread = self._thread
        if server is not None:
            server.should_exit = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
        self._teardown()

    def _teardown(self) -> None:
        self._url = None
        self._thread = None
        self._uvicorn_server = None


__all__ = ["HarnessMCPBridge"]
