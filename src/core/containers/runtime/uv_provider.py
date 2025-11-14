"""Providers for launching Hugging Face Spaces via ``uv run``."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Dict, Optional

import requests

from .providers import RuntimeProvider


def _check_uv_installed() -> None:
    try:
        subprocess.check_output(["uv", "--version"])
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`uv` executable not found. Install uv from https://docs.astral.sh and ensure it is on PATH."
        ) from exc


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.listen(1)
        return sock.getsockname()[1]
   

def _create_uv_command(
    repo_id: str,
    port: int,
    reload: bool,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--isolated",
        "--with",
        f"git+https://huggingface.co/spaces/{repo_id}",
        "--",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")
    return command


def _poll_health(health_url: str, timeout_s: float) -> None:
    """Poll a health endpoint until it returns HTTP 200 or times out."""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            timeout = max(0.0001, min(deadline - time.time(), 2.0))
            response = requests.get(health_url, timeout=timeout)
            if response.status_code == 200:
                return
        except requests.RequestException:
            continue

        time.sleep(0.5)

    raise TimeoutError(f"Server did not become ready within {timeout_s:.1f} seconds")


class UVProvider(RuntimeProvider):
    """
    RuntimeProvider implementation backed by ``uv run``.

    Args:
        repo_id: The repository ID of the environment to run
        reload: Whether to reload the environment on code changes
        env_vars: Environment variables to pass to the environment
        context_timeout_s: The timeout to wait for the environment to become ready

    Example:
        >>> provider = UVProvider(repo_id="burtenshaw/echo-cli")
        >>> base_url = provider.start()
        >>> print(base_url)  # http://localhost:8000
        >>> # Use the environment via base_url
        >>> provider.stop()
    """

    def __init__(
        self,
        repo_id: str,
        reload: bool = False,
        env_vars: Optional[Dict[str, str]] = None,
        context_timeout_s: float = 60.0,
    ):
        """Initialize the UVProvider."""
        self.repo_id = repo_id
        self.reload = reload
        self.env_vars = env_vars
        self.context_timeout_s = context_timeout_s
        _check_uv_installed()
        self._process = None
        self._base_url = None

    def start(
        self,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **_: Dict[str, str],
    ) -> str:
        """
        Start the environment via `uv run`.

        Args:
            port: The port to bind the environment to
            env_vars: Environment variables to pass to the environment

        Returns:
            The base URL of the environment

        Raises:
            RuntimeError: If the environment is already running
        """
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("UVProvider is already running")

        bind_port = port or _find_free_port()

        command = _create_uv_command(
            repo_id=self.repo_id,
            port=bind_port,
            reload=self.reload,
        )

        env = os.environ.copy()

        if self.env_vars:
            env.update(self.env_vars)
        if env_vars:
            env.update(env_vars)

        try:
            self._process = subprocess.Popen(command, env=env)
        except OSError as exc:
            raise RuntimeError(f"Failed to launch `uv run`: {exc}") from exc

        self._base_url = f"http://localhost:{bind_port}"
        return self._base_url

    def wait_for_ready(self, timeout_s: float = 60.0) -> None:
        """
        Wait for the environment to become ready.

        Args:
            timeout_s: The timeout to wait for the environment to become ready

        Raises:
            RuntimeError: If the environment is not running
            TimeoutError: If the environment does not become ready within the timeout
        """
        if self._process and self._process.poll() is not None:
            code = self._process.returncode
            raise RuntimeError(f"uv process exited prematurely with code {code}")

        _poll_health(f"{self._base_url}/health", timeout_s=timeout_s)

    def stop(self) -> None:
        """
        Stop the environment.

        Raises:
            RuntimeError: If the environment is not running
        """
        if self._process is None:
            return

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)

        self._process = None
        self._base_url = None

    @property
    def base_url(self) -> str:
        """
        The base URL of the environment.

        Returns:
            The base URL of the environment

        Raises:
            RuntimeError: If the environment is not running
        """
        if self._base_url is None:
            raise RuntimeError("UVProvider has not been started")
        return self._base_url
