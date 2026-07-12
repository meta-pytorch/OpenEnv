# SPDX-License-Identifier: BSD-3-Clause

"""Tests for validation certification trust-boundary enforcement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openenv.core.containers.runtime import hf_sandbox_provider as provider_module
from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider
from openenv.validation.security import ensure_official_hf_sandbox


def test_official_certification_rejects_pooled_hf_sandbox() -> None:
    pooled = HFSandboxProvider(image="example")
    dedicated = HFSandboxProvider(image="example", mode="dedicated")

    with pytest.raises(RuntimeError, match="dedicated"):
        ensure_official_hf_sandbox(pooled)

    ensure_official_hf_sandbox(dedicated)


@pytest.mark.parametrize(
    "env_vars",
    [
        {"HF_TOKEN": "secret"},
        {"HF_AUTH": "hf_1234567890abcdef"},
        {"AUTH": "Bearer coordinator-token"},
        {"AWS_ACCESS_KEY_ID": "AKIA1234567890ABCDEF"},
        {"MODE": "ASIA1234567890ABCDEF"},
        {"DATABASE_URL": "https://alice:hunter2@example.com/database"},
        {"DATABASE_URL": "postgresql://alice:hunter2@db/app"},
        {"CACHE_URL": "redis://:hunter2@cache/0"},
        {"PGPASSWORD": "secret"},
        {"MYSQL_PWD": "secret"},
        {"CI_JOB_JWT": "secret"},
        {"GIT_ASKPASS": "/tmp/helper"},
        {"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/google.json"},
        {"NETRC": "/tmp/netrc"},
        {"MODE": "postgresql://db/app?password=hunter2"},
        {"MODE": "redis://cache/0?token=hunter2"},
        {"MODE": "https://service.example/path?api_key=hunter2"},
        {
            "DOWNLOAD_URL": (
                "https://service.example/object?X-Amz-Credential=scope&"
                "X-Amz-Signature=deadbeef"
            )
        },
        {
            "DOWNLOAD_URL": (
                "https://service.example/object?X-Goog-Credential=scope&"
                "X-Goog-Signature=deadbeef"
            )
        },
        {"MODE": "hf_1234567890abcdef"},
    ],
)
def test_official_certification_rejects_forwarded_credentials(
    env_vars: dict[str, str],
) -> None:
    provider = HFSandboxProvider(
        image="example",
        mode="dedicated",
        env_vars=env_vars,
    )

    with pytest.raises(RuntimeError, match="credential-like"):
        ensure_official_hf_sandbox(provider)


def test_official_certification_locks_start_environment() -> None:
    provider = HFSandboxProvider(
        image="example",
        mode="dedicated",
        env_vars={"MODE": "certify"},
    )
    ensure_official_hf_sandbox(provider)

    with pytest.raises(RuntimeError, match="environment overrides"):
        provider.start_container(env_vars={"HF_TOKEN": "secret"})


def test_official_certification_detects_post_lock_environment_mutation() -> None:
    provider = HFSandboxProvider(
        image="example",
        mode="dedicated",
        env_vars={"MODE": "certify"},
    )
    ensure_official_hf_sandbox(provider)
    assert provider._env_vars is not None
    provider._env_vars["MODE"] = "changed"

    with pytest.raises(RuntimeError, match="execution-setting changes"):
        provider.start_container()


def test_official_start_rejects_unverifiable_runtime_isolation() -> None:
    class SandboxWithoutHostIdentity:
        proxy_headers: dict[str, str] = {}

        def run(self, *args: object, **kwargs: object) -> MagicMock:
            return MagicMock()

        def proxy_url_for(self, port: int, path: str) -> str:
            return "https://sandbox.example/proxy"

        def kill(self) -> None:
            self._killed = True

    sandbox = SandboxWithoutHostIdentity()
    sandbox_cls = MagicMock()
    sandbox_cls.create.return_value = sandbox
    provider = HFSandboxProvider(image="example", mode="dedicated")
    ensure_official_hf_sandbox(provider)

    with patch.object(provider_module, "_get_sandbox_cls", return_value=sandbox_cls):
        with pytest.raises(RuntimeError, match="verify dedicated isolation"):
            provider.start_container()

    assert provider._sandbox is None


def test_official_certification_cannot_lock_active_sandbox() -> None:
    provider = HFSandboxProvider(image="example", mode="dedicated")
    sandbox = MagicMock()
    sandbox.host_id = None
    provider._sandbox = sandbox

    with pytest.raises(RuntimeError, match="before.*started"):
        ensure_official_hf_sandbox(provider)

    assert provider._official_certification is False
