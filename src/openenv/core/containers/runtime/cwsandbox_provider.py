"""CoreWeave Sandbox provider for running OpenEnv environments."""

from __future__ import annotations

import os
import shlex
import socket
import ssl
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .providers import ContainerProvider


class CWSandboxProvider(ContainerProvider):
    """Container provider that runs OpenEnv servers in CoreWeave Sandbox.

    The cwsandbox SDK uses a keepalive command by default, so this provider starts
    the OpenEnv server explicitly after the sandbox reaches RUNNING.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        ingress_mode: Optional[str] = "public",
        egress_mode: Optional[str] = "internet",
        resources: Optional[Any] = None,
        request_timeout_seconds: float = 300.0,
        max_lifetime_seconds: Optional[float] = None,
        max_timeout_seconds: Optional[int] = None,
        tags: Optional[list[str]] = None,
        profile_ids: Optional[list[str]] = None,
        profile_names: Optional[list[str]] = None,
        runner_ids: Optional[list[str]] = None,
        cmd: Optional[str] = None,
        url_scheme: str = "http",
        delete_on_stop: bool = True,
        ready_settle_seconds: float = 2.0,
        sdk: Optional[Any] = None,
    ) -> None:
        self._sdk = sdk if sdk is not None else _import_cwsandbox()
        self._base_url = base_url
        self._ingress_mode = ingress_mode
        self._egress_mode = egress_mode
        self._resources = resources
        self._request_timeout_seconds = request_timeout_seconds
        self._max_lifetime_seconds = max_lifetime_seconds
        self._max_timeout_seconds = max_timeout_seconds
        self._tags = tags
        self._profile_ids = profile_ids
        self._profile_names = profile_names
        self._runner_ids = runner_ids
        self._cmd = cmd
        self._url_scheme = url_scheme.rstrip(":/")
        self._delete_on_stop = delete_on_stop
        self._ready_settle_seconds = ready_settle_seconds

        self._sandbox: Any = None
        self._base_env_url: Optional[str] = None

    @classmethod
    def preflight(cls, *, sdk: Any | None = None) -> None:
        """Validate that CoreWeave Sandbox auth works before launch."""
        resolved_sdk = sdk if sdk is not None else _import_cwsandbox()
        if sdk is None and not os.environ.get("CWSANDBOX_API_KEY"):
            raise SystemExit(
                "CoreWeave Sandbox requires CWSANDBOX_API_KEY to be set. "
                "Please set this environment variable and try again."
            )
        try:
            resolved_sdk.Sandbox.list().result()
        except Exception as exc:
            auth_error = getattr(resolved_sdk, "CWSandboxAuthenticationError", None)
            if isinstance(auth_error, type) and isinstance(exc, auth_error):
                raise SystemExit(
                    f"CoreWeave Sandbox auth check failed: {exc}. "
                    "Verify your CWSANDBOX_API_KEY and try again."
                ) from exc
            raise

    @property
    def base_url(self) -> Optional[str]:
        """Base URL returned to OpenEnv clients after startup."""
        return self._base_env_url

    @property
    def sandbox_id(self) -> Optional[str]:
        """Sandbox ID for debugging after startup."""
        if self._sandbox is None:
            return None
        return getattr(self._sandbox, "sandbox_id", None)

    @property
    def service_address(self) -> Optional[str]:
        """External service address reported by cwsandbox after startup."""
        if self._sandbox is None:
            return None
        return getattr(self._sandbox, "service_address", None)

    def start_container(
        self,
        image: str,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """Start an OpenEnv image in a sandbox and return its base URL."""
        port = port or 8000
        if port != 8000:
            raise ValueError(
                f"CWSandboxProvider only supports port 8000 (got {port}). "
                "OpenEnv images and generated server commands listen on port 8000."
            )

        cmd = kwargs.pop("cmd", None) or self._cmd
        unknown = set(kwargs)
        if unknown:
            raise ValueError(f"Unsupported kwargs for CWSandboxProvider: {unknown}")

        network = self._sdk.NetworkOptions(
            ingress_mode=self._ingress_mode,
            exposed_ports=(port,),
            egress_mode=self._egress_mode,
        )

        run_kwargs: dict[str, Any] = {
            "container_image": image,
            "network": network,
            "request_timeout_seconds": self._request_timeout_seconds,
        }
        optional_kwargs = {
            "base_url": self._base_url,
            "environment_variables": env_vars,
            "resources": self._resources,
            "max_lifetime_seconds": self._max_lifetime_seconds,
            "max_timeout_seconds": self._max_timeout_seconds,
            "tags": self._tags,
            "profile_ids": self._profile_ids,
            "profile_names": self._profile_names,
            "runner_ids": self._runner_ids,
        }
        run_kwargs.update(
            {key: value for key, value in optional_kwargs.items() if value is not None}
        )

        sandbox = self._sdk.Sandbox.run(**run_kwargs)
        self._sandbox = sandbox
        try:
            sandbox.wait(timeout=self._request_timeout_seconds)
            server_cmd = cmd or self._discover_server_cmd(sandbox, port=port)
            self._start_server(sandbox, server_cmd)
            self._base_env_url = self._base_url_from_sandbox(sandbox)
            return self._base_env_url
        except Exception:
            self.stop_container()
            raise

    def stop_container(self) -> None:
        """Stop and optionally delete the sandbox."""
        sandbox = self._sandbox
        self._sandbox = None
        self._base_env_url = None
        if sandbox is None:
            return

        sandbox_id = getattr(sandbox, "sandbox_id", None)
        try:
            self._result_with_retries(lambda: sandbox.stop(missing_ok=True))
        finally:
            if self._delete_on_stop and sandbox_id:
                self._result_with_retries(
                    lambda: self._sdk.Sandbox.delete(
                        sandbox_id,
                        base_url=self._base_url,
                        timeout_seconds=self._request_timeout_seconds,
                        missing_ok=True,
                    )
                )

    def wait_for_ready(self, base_url: str, timeout_s: float = 120.0) -> None:
        """Wait for the OpenEnv server to respond to /health and /ws."""
        import requests

        deadline = time.time() + timeout_s
        health_url = f"{base_url.rstrip('/')}/health"
        last_error: Optional[BaseException] = None

        while time.time() < deadline:
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200 and self._websocket_ready(base_url):
                    if self._ready_settle_seconds:
                        time.sleep(self._ready_settle_seconds)
                    return
            except requests.RequestException as exc:
                last_error = exc

            if self._server_process_dead():
                log = self._server_log()
                raise RuntimeError(f"OpenEnv server process exited.\nLog:\n{log}")

            time.sleep(1.0)

        detail = f" Last error: {last_error}" if last_error else ""
        raise TimeoutError(
            f"OpenEnv server at {base_url} did not become ready within {timeout_s}s."
            f"{detail}"
        )

    @staticmethod
    def _websocket_ready(base_url: str, timeout: float = 5.0) -> bool:
        """Return True once the ingress path accepts a /ws upgrade."""
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False

        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        # RFC 6455 sample nonce; this is a public probe value, not a secret.
        websocket_key = "dGhlIHNhbXBsZSBub25jZQ=="
        request = (
            "GET /ws HTTP/1.1\r\n"
            f"Host: {parsed.netloc}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {websocket_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode()

        try:
            with socket.create_connection(
                (parsed.hostname, port), timeout=timeout
            ) as raw:
                if parsed.scheme == "https":
                    context = ssl.create_default_context()
                    with context.wrap_socket(
                        raw, server_hostname=parsed.hostname
                    ) as sock:
                        sock.settimeout(timeout)
                        sock.sendall(request)
                        return b" 101 " in sock.recv(256)
                raw.settimeout(timeout)
                raw.sendall(request)
                return b" 101 " in raw.recv(256)
        except OSError:
            return False

    def _discover_server_cmd(self, sandbox: Any, port: int) -> str:
        yaml_path = self._find_openenv_yaml(sandbox)
        if yaml_path is None:
            raise ValueError(
                "Could not find openenv.yaml in the sandbox. Pass cmd= to "
                "CWSandboxProvider or start_container()."
            )

        content = self._exec_stdout(
            sandbox,
            f"cat {shlex.quote(yaml_path)}",
            timeout_seconds=10,
        )
        app = self._parse_app_field(content)
        if app is None:
            raise ValueError(
                f"openenv.yaml at {yaml_path} does not contain an 'app' field. "
                "Pass cmd= to CWSandboxProvider or start_container()."
            )

        env_root = yaml_path.rsplit("/", 1)[0]
        return (
            f"cd {shlex.quote(env_root)} && "
            f"python -m uvicorn {shlex.quote(app)} --host 0.0.0.0 --port {port}"
        )

    def _find_openenv_yaml(self, sandbox: Any) -> Optional[str]:
        candidate = "/app/env/openenv.yaml"
        out = self._exec_stdout(
            sandbox,
            f"test -f {shlex.quote(candidate)} && echo found",
            timeout_seconds=10,
        )
        if "found" in out:
            return candidate

        path = self._exec_stdout(
            sandbox,
            "find /app -maxdepth 5 -name openenv.yaml -print -quit 2>/dev/null",
            timeout_seconds=10,
        ).strip()
        if path.startswith("/"):
            return path
        return None

    @staticmethod
    def _parse_app_field(yaml_content: str) -> Optional[str]:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "CWSandboxProvider command discovery requires PyYAML. "
                "Install openenv with its normal dependencies."
            ) from exc

        try:
            data = yaml.safe_load(yaml_content) or {}
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        value = data.get("app")
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return None

    def _start_server(self, sandbox: Any, cmd: str) -> None:
        escaped = shlex.quote(cmd)
        self._exec_stdout(
            sandbox,
            f"nohup bash -c {escaped} > /tmp/openenv-server.log 2>&1 & "
            "echo $! > /tmp/openenv-server.pid",
            timeout_seconds=10,
        )

    def _base_url_from_sandbox(self, sandbox: Any) -> str:
        address = getattr(sandbox, "service_address", None)
        if not address:
            raise RuntimeError(
                "Sandbox did not return service_address; cannot construct OpenEnv URL."
            )
        if address.startswith(("http://", "https://")):
            return address.rstrip("/")
        return f"{self._url_scheme_from_sandbox(sandbox)}://{address}".rstrip("/")

    def _url_scheme_from_sandbox(self, sandbox: Any) -> str:
        exposed_ports = getattr(sandbox, "exposed_ports", None) or ()
        for _port, name in exposed_ports:
            if name in {"http", "https"}:
                return name
        return self._url_scheme

    def _server_process_dead(self) -> bool:
        sandbox = self._sandbox
        if sandbox is None:
            return False
        out = self._exec_stdout(
            sandbox,
            "test -f /tmp/openenv-server.pid && "
            "kill -0 $(cat /tmp/openenv-server.pid) 2>/dev/null "
            "&& echo RUNNING || echo DEAD",
            timeout_seconds=10,
            suppress_errors=True,
        )
        return "DEAD" in out

    def _server_log(self) -> str:
        sandbox = self._sandbox
        if sandbox is None:
            return ""
        return self._exec_stdout(
            sandbox,
            "cat /tmp/openenv-server.log 2>/dev/null",
            timeout_seconds=10,
            suppress_errors=True,
        )

    @staticmethod
    def _exec_stdout(
        sandbox: Any,
        script: str,
        *,
        timeout_seconds: int,
        suppress_errors: bool = False,
    ) -> str:
        try:
            process = sandbox.exec(
                ["bash", "-lc", script],
                timeout_seconds=timeout_seconds,
            )
            result = process.result()
            return getattr(result, "stdout", "") or ""
        except Exception:
            if suppress_errors:
                return ""
            raise

    @staticmethod
    def _result_with_retries(operation: Any, attempts: int = 3) -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(attempts):
            try:
                return operation().result()
            except Exception as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                time.sleep(1.0)
        assert last_error is not None
        raise last_error


def _import_cwsandbox() -> Any:
    try:
        return __import__("cwsandbox")
    except ImportError as exc:
        raise RuntimeError(
            "CWSandboxProvider requires cwsandbox. Install it with "
            "`pip install cwsandbox`."
        ) from exc
