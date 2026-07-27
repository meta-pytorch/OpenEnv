# SPDX-License-Identifier: BSD-3-Clause

"""
Fystash container provider for running OpenEnv environment servers in warm
Firecracker rooms (Loop 94 first cut).

Requires ``FYSTASH_API_KEY``. Optional: ``FYSTASH_API`` (default
``https://api.fystash.ai``), ``FYSTASH_TEMPLATE_ID`` (default ``docker``).

Honesty / v1 limits:

- Accepts a **registry image** string only — no ``image_from_dockerfile`` /
  snapshot build (unlike Daytona/Modal).
- Starts ``template_id=docker``, ``docker pull`` + ``docker run`` inside the
  guest, then exposes port 8000 via Fystash preview URL (Loop 47).
- WebSocket / ``/ws`` proxy behaviour depends on the preview path; treat as
  experimental until validated against a full EnvClient session.

Docs: https://fystash.ai · https://docs.fystash.ai
"""

from __future__ import annotations

import base64
import hashlib
import os
import shlex
import time
import uuid
from typing import Any, Dict, Optional

import httpx

from .providers import ContainerProvider


class _FystashApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Fystash API HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class _FystashApi:
    """Minimal sync Room API client (httpx). No unpublished PyPI package dep."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 300.0) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        resp = self._client.request(method, path, json=json_body, params=params)
        if resp.status_code >= 400:
            detail = resp.text
            try:
                parsed = resp.json()
                detail = str(parsed.get("detail", detail))
            except Exception:  # noqa: BLE001
                pass
            raise _FystashApiError(resp.status_code, detail)
        if not resp.content:
            return {}
        return resp.json()

    def create_room(self, room_id: str) -> dict[str, Any]:
        return self._request("POST", "/v1/rooms", json_body={"room_id": room_id})

    def create_from_template(
        self,
        room_id: str,
        agent_id: str,
        *,
        guest_cid: int,
        template_id: str,
        memory_mib: int,
        vcpu_count: int = 2,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/rooms/{room_id}/sandboxes/from-template",
            json_body={
                "agent_id": agent_id,
                "guest_cid": guest_cid,
                "template_id": template_id,
                "vcpu_count": vcpu_count,
                "memory_mib": memory_mib,
                "enable_guest_net": True,
                "attach_fabric": True,
            },
        )

    def exec(
        self,
        room_id: str,
        agent_id: str,
        argv: list[str],
        *,
        timeout_ms: int = 300_000,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"argv": argv, "timeout_ms": timeout_ms}
        if env:
            body["env"] = env
        return self._request(
            "POST",
            f"/v1/rooms/{room_id}/sandboxes/{agent_id}/exec",
            json_body=body,
        )

    def expose_port(
        self, room_id: str, agent_id: str, port: int
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/rooms/{room_id}/sandboxes/{agent_id}/ports",
            json_body={"port": port},
        )

    def destroy(self, room_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/rooms/{room_id}")


def _sanitize_room_id(prefix: str = "oe") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _guest_cid(seed: str) -> int:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return 9400 + (int(digest[:8], 16) % 500)


def _decode_b64(value: Any) -> str:
    raw = value or b""
    if isinstance(raw, str):
        raw = raw.encode()
    try:
        return base64.b64decode(raw).decode(errors="replace")
    except Exception:  # noqa: BLE001
        return ""


class FystashProvider(ContainerProvider):
    """
    Container provider that runs OpenEnv servers in Fystash Firecracker rooms.

    Example:
        >>> provider = FystashProvider()
        >>> base_url = provider.start_container("ghcr.io/example/echo-env:latest")
        >>> provider.wait_for_ready(base_url)
        >>> provider.stop_container()
    """

    def __init__(
        self,
        *,
        image: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        template_id: Optional[str] = None,
        agent_id: str = "openenv",
        memory_mib: int = 2048,
        cmd: Optional[str] = None,
        create_timeout: float = 600.0,
    ):
        """
        Args:
            image: Registry image used when ``start_container()`` omits *image*.
            env_vars: Default env vars for ``docker run``.
            api_key: Fystash org API key (or ``FYSTASH_API_KEY``).
            api_url: Control plane URL (or ``FYSTASH_API``, default public API).
            template_id: Fystash template (default ``docker`` / ``FYSTASH_TEMPLATE_ID``).
            agent_id: Sandbox agent id inside the room.
            memory_mib: Guest memory for the docker template path.
            cmd: Optional shell command overriding the image CMD.
            create_timeout: Seconds budget for pull + run + expose (advisory).
        """
        self._api_key = api_key or os.environ.get("FYSTASH_API_KEY") or ""
        self._api_url = (
            api_url
            or os.environ.get("FYSTASH_API")
            or "https://api.fystash.ai"
        ).rstrip("/")
        self._template_id = (
            template_id
            or os.environ.get("FYSTASH_TEMPLATE_ID")
            or "docker"
        )
        self._image = image
        self._env_vars = env_vars
        self._agent_id = agent_id
        self._memory_mib = memory_mib
        self._cmd = cmd
        self._create_timeout = create_timeout

        self._api: _FystashApi | None = None
        self._room_id: str | None = None
        self._preview_url: str | None = None
        self._base_url: str | None = None

    def start_container(
        self,
        image: Optional[str] = None,
        port: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create a Fystash room, run *image* under dockerd, expose preview URL.

        Args:
            image: Registry image (e.g. ``"echo-env:latest"``). ``dockerfile:``
                and ``snapshot:`` prefixes are not supported in v1.
            port: Must be ``None`` or ``8000`` (preview maps this port).
            env_vars: Environment variables for ``docker run``.
            **kwargs: ``cmd`` (str) overrides the image command.

        Returns:
            HTTPS preview base URL for the OpenEnv server.
        """
        if not self._api_key:
            raise RuntimeError(
                "FystashProvider requires FYSTASH_API_KEY. "
                "Signup: https://fystash.ai/signup"
            )

        if port is not None and port != 8000:
            raise ValueError(
                f"FystashProvider only supports port 8000 (got {port}). "
                "OpenEnv servers listen on 8000; Fystash preview exposes that port."
            )

        effective_image = image if image is not None else self._image
        if effective_image is None:
            raise ValueError(
                "FystashProvider requires an image. Pass it to the constructor "
                "or start_container()."
            )
        if effective_image.startswith(("dockerfile:", "snapshot:")):
            raise ValueError(
                "FystashProvider v1 does not build Dockerfiles or snapshots. "
                "Pass a registry image tag (e.g. after `openenv build` / docker push)."
            )

        effective_env = self._env_vars if env_vars is None else env_vars
        cmd = kwargs.pop("cmd", None) or self._cmd
        if kwargs:
            # Ignore unknown kwargs for forward-compat with other providers.
            pass

        api = _FystashApi(self._api_url, self._api_key, timeout=self._create_timeout)
        self._api = api
        room_id = _sanitize_room_id("oe")
        self._room_id = room_id
        guest_cid = _guest_cid(room_id)

        try:
            api.create_room(room_id)
            api.create_from_template(
                room_id,
                self._agent_id,
                guest_cid=guest_cid,
                template_id=self._template_id,
                memory_mib=self._memory_mib,
            )

            self._wait_docker_ready(api, room_id)
            self._docker_pull(api, room_id, effective_image)
            self._docker_run(
                api,
                room_id,
                effective_image,
                env_vars=effective_env,
                cmd=cmd,
            )

            exposed = api.expose_port(room_id, self._agent_id, 8000)
            url = str(exposed.get("url") or "").rstrip("/")
            if not url:
                raise RuntimeError(
                    f"expose_port returned no url: {exposed!r}"
                )
            self._preview_url = url
            self._base_url = url
            return url
        except Exception:
            self.stop_container()
            raise

    def _exec(
        self,
        api: _FystashApi,
        room_id: str,
        command: str,
        *,
        timeout_ms: int = 300_000,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        resp = api.exec(
            room_id,
            self._agent_id,
            ["/bin/bash", "-lc", command],
            timeout_ms=timeout_ms,
            env=env,
        )
        code = int(resp["exit_code"]) if resp.get("exit_code") is not None else 1
        return (
            code,
            _decode_b64(resp.get("stdout_b64")),
            _decode_b64(resp.get("stderr_b64")),
        )

    def _wait_docker_ready(self, api: _FystashApi, room_id: str) -> None:
        deadline = time.time() + min(120.0, self._create_timeout)
        last = ""
        while time.time() < deadline:
            code, out, err = self._exec(
                api, room_id, "docker info >/dev/null 2>&1", timeout_ms=30_000
            )
            if code == 0:
                return
            last = err or out
            time.sleep(2.0)
        raise RuntimeError(
            "dockerd not ready in Fystash docker template within timeout. "
            f"Last output: {last[:500]}"
        )

    def _docker_pull(self, api: _FystashApi, room_id: str, image: str) -> None:
        code, out, err = self._exec(
            api,
            room_id,
            f"docker pull {shlex.quote(image)}",
            timeout_ms=int(self._create_timeout * 1000),
        )
        if code != 0:
            raise RuntimeError(
                f"docker pull failed for {image!r}: {(err or out)[:800]}"
            )

    def _docker_run(
        self,
        api: _FystashApi,
        room_id: str,
        image: str,
        *,
        env_vars: dict[str, str] | None,
        cmd: str | None,
    ) -> None:
        env_flags = ""
        if env_vars:
            for key, value in env_vars.items():
                env_flags += (
                    f" -e {shlex.quote(f'{key}={value}')}"
                )
        # Publish 8000 on the guest; Fystash preview proxies to guest:8000.
        run_cmd = (
            "docker rm -f openenv-server >/dev/null 2>&1 || true; "
            f"docker run -d --name openenv-server -p 8000:8000{env_flags} "
        )
        if cmd:
            # Override image CMD with an explicit shell command.
            run_cmd += (
                f"{shlex.quote(image)} bash -lc {shlex.quote(cmd)}"
            )
        else:
            run_cmd += shlex.quote(image)

        code, out, err = self._exec(
            api, room_id, run_cmd, timeout_ms=180_000
        )
        if code != 0:
            raise RuntimeError(
                f"docker run failed for {image!r}: {(err or out)[:800]}"
            )

    def stop_container(self) -> None:
        """Destroy the Fystash room (and preview routes)."""
        api = self._api
        room_id = self._room_id
        try:
            if api is not None and room_id is not None:
                try:
                    api.destroy(room_id)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._room_id = None
            self._preview_url = None
            self._base_url = None
            if api is not None:
                try:
                    api.close()
                except Exception:  # noqa: BLE001
                    pass
            self._api = None

    def close(self) -> None:
        self.stop_container()

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError(
                "FystashProvider has no active base_url. Start the provider "
                "before reading base_url."
            )
        return self._base_url

    def wait_for_ready(self, base_url: str, timeout_s: float = 180.0) -> None:
        """
        Poll ``{base_url}/health`` until HTTP 200.

        Uses a longer default timeout than local Docker because create + pull
        can be slow on nested-KVM.
        """
        import requests

        url = base_url.rstrip("/") + "/health"
        deadline = time.time() + timeout_s
        last_err = ""
        while time.time() < deadline:
            try:
                response = requests.get(url, timeout=5.0)
                if response.status_code == 200:
                    return
                last_err = f"status={response.status_code}"
            except requests.RequestException as exc:
                last_err = str(exc)
            time.sleep(2.0)
        raise TimeoutError(
            f"Fystash OpenEnv server not ready at {url} within {timeout_s}s "
            f"({last_err})"
        )
