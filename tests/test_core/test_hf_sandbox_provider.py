# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Hugging Face sandbox isolation modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime import hf_sandbox_provider as provider_module
from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider
from starlette.testclient import TestClient


def _proxy() -> MagicMock:
    proxy = MagicMock()
    proxy.start.return_value = "http://127.0.0.1:12345"
    return proxy


def test_hf_sandbox_provider_defaults_to_pooled_mode() -> None:
    sandbox = MagicMock()
    sandbox.proxy_url_for.return_value = "https://sandbox.example/proxy"
    sandbox.proxy_headers = {"Authorization": "Bearer token"}
    pool = MagicMock()
    pool.create.return_value = sandbox
    proxy = _proxy()

    with patch.object(provider_module, "_get_pool", return_value=pool) as get_pool:
        with patch.object(
            provider_module, "_LocalAuthProxy", return_value=proxy
        ) as proxy_cls:
            instance = HFSandboxProvider(
                image="hf.co/spaces/openenv/coding_env", env_vars={"MODE": "test"}
            )
            base_url = instance.start_container()

    assert instance.mode == "pooled"
    assert base_url == "http://127.0.0.1:12345"
    get_pool.assert_called_once_with("hf.co/spaces/openenv/coding_env", "cpu-basic")
    pool.create.assert_called_once_with(env={"MODE": "test"})
    command = sandbox.run.call_args.args[0]
    assert "SBX_PROXY_DIR" in command
    assert sandbox.run.call_args.kwargs == {"shell": True, "background": True}
    proxy_cls.assert_called_once_with(
        target_url="https://sandbox.example/proxy",
        headers={"Authorization": "Bearer token"},
    )


def test_hf_sandbox_provider_dedicated_mode_uses_sandbox_create() -> None:
    sandbox = MagicMock()
    sandbox.proxy_url_for.return_value = "https://sandbox.example/proxy"
    sandbox.proxy_headers = {}
    sandbox_cls = MagicMock()
    sandbox_cls.create.return_value = sandbox
    proxy = _proxy()

    with patch.object(
        provider_module, "_get_sandbox_cls", return_value=sandbox_cls
    ) as get_sandbox_cls:
        with patch.object(provider_module, "_get_pool") as get_pool:
            with patch.object(provider_module, "_LocalAuthProxy", return_value=proxy):
                instance = HFSandboxProvider(
                    image="hf.co/spaces/openenv/echo",
                    flavor="a10g-small",
                    env_vars={"MODE": "certify"},
                    mode="dedicated",
                )
                instance.start_container()

    get_sandbox_cls.assert_called_once_with()
    get_pool.assert_not_called()
    sandbox_cls.create.assert_called_once_with(
        image="hf.co/spaces/openenv/echo",
        flavor="a10g-small",
        env={"MODE": "certify"},
    )
    command = sandbox.run.call_args.args[0]
    assert "unset SBX_PROXY_DIR" in command
    assert sandbox.run.call_args.kwargs == {"shell": True, "background": True}


def test_hf_sandbox_provider_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        HFSandboxProvider(image="example", mode="shared")  # type: ignore[arg-type]


def test_hf_sandbox_provider_requires_dedicated_mode_for_secrets() -> None:
    with pytest.raises(ValueError, match="dedicated"):
        HFSandboxProvider(image="example", secrets={"API_KEY": "secret"})


def test_hf_sandbox_provider_uses_encrypted_dedicated_secrets() -> None:
    sandbox_cls = MagicMock()
    provider = HFSandboxProvider(
        image="example",
        mode="dedicated",
        secrets={"API_KEY": "secret"},
    )

    with patch.object(provider_module, "_get_sandbox_cls", return_value=sandbox_cls):
        provider._create_sandbox(image="example", env_vars={"MODE": "test"})

    sandbox_cls.create.assert_called_once_with(
        image="example",
        flavor="cpu-basic",
        env={"MODE": "test"},
        secrets={"API_KEY": "secret"},
    )


def test_hf_sandbox_provider_mode_is_read_only() -> None:
    provider = HFSandboxProvider(image="example", mode="dedicated")

    with pytest.raises(AttributeError):
        provider.mode = "pooled"  # type: ignore[misc]


def test_stop_retains_sandbox_handle_when_kill_fails() -> None:
    provider = HFSandboxProvider(image="example", mode="dedicated")
    sandbox = MagicMock()
    sandbox.kill.side_effect = RuntimeError("cancel failed")
    provider._sandbox = sandbox

    with pytest.raises(RuntimeError, match="cancel failed"):
        provider.stop_container()

    assert provider._sandbox is sandbox


def test_stop_retains_sandbox_when_sdk_swallows_kill_failure() -> None:
    provider = HFSandboxProvider(image="example", mode="dedicated")
    sandbox = MagicMock()
    sandbox._killed = False
    provider._sandbox = sandbox

    with pytest.raises(RuntimeError, match="did not confirm termination"):
        provider.stop_container()

    assert provider._sandbox is sandbox


def test_stop_fails_closed_without_sdk_termination_state() -> None:
    class SandboxWithoutTerminationState:
        def kill(self) -> None:
            return None

    provider = HFSandboxProvider(image="example", mode="dedicated")
    sandbox = SandboxWithoutTerminationState()
    provider._sandbox = sandbox

    with pytest.raises(RuntimeError, match="did not confirm termination"):
        provider.stop_container()

    assert provider._sandbox is sandbox


def test_local_auth_proxy_rejects_upstream_redirects() -> None:
    class _ServerWithoutSocket:
        def __init__(self, config: object):
            self.config = config
            self.started = True
            self.should_exit = False

        def run(self) -> None:
            return None

    upstream = MagicMock()
    upstream.status_code = 302
    upstream.headers = {"Location": "http://169.254.169.254/latest/meta-data"}

    with patch.object(provider_module, "_find_available_port", return_value=12345):
        with patch.object(provider_module.uvicorn, "Server", _ServerWithoutSocket):
            proxy = provider_module._LocalAuthProxy(
                target_url="https://sandbox.example/proxy",
                headers={"X-Sandbox-Token": "secret"},
            )
            proxy.start()
            assert proxy._server is not None
            app = proxy._server.config.app
            with patch.object(
                provider_module.requests, "request", return_value=upstream
            ):
                response = TestClient(app).get("/health", follow_redirects=False)
            proxy.stop()

    assert response.status_code == 502
    assert response.headers.get("Location") is None
    upstream.close.assert_called_once_with()
