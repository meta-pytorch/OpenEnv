# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for author-triggered validation in a dedicated HF Sandbox."""

from __future__ import annotations

import importlib
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import Sandbox, SandboxPool
from openenv.validation.models import (
    RunnerCapabilities,
    ValidationCapability,
    ValidationProfile,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)

from ._helpers import write_valid_env


_REVISION = "a" * 40
_SANDBOX_IMAGE = "python:3.12"


def _remote_module() -> ModuleType:
    """Load the API under test while keeping this test file collectable in red."""
    module_name = "openenv.validation.remote"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        pytest.fail(
            "missing implementation: openenv.validation.remote must provide "
            "run_remote_validation and RemoteValidationError"
        )
    for attribute in ("run_remote_validation", "RemoteValidationError"):
        if not hasattr(module, attribute):
            pytest.fail(f"missing implementation: {module_name}.{attribute}")
    return module


def _report_payload(target: Path) -> dict[str, object]:
    result = ValidationResult(
        criterion_id="source.validation_spec",
        requirement="Detect the validation spec",
        status=ValidationStatus.PASS,
        severity=ValidationSeverity.BLOCKING,
        evidence={"state": "loaded"},
        duration_s=0.01,
        timeout_s=10.0,
        required_capabilities=frozenset({ValidationCapability.SOURCE}),
    )
    report = ValidationReport(
        target=str(target),
        profile=ValidationProfile.PUBLISH,
        policy_version="rfc008-v1",
        runner=RunnerCapabilities(
            runner="hf-sandbox",
            available=frozenset(
                {ValidationCapability.SOURCE, ValidationCapability.RUNTIME}
            ),
            official=False,
            isolation_mode="dedicated",
        ),
        results=(result,),
        duration_s=0.1,
        started_at="2026-07-12T12:00:00+00:00",
        finished_at="2026-07-12T12:00:00.100000+00:00",
        repo_sha=_REVISION,
        certified=False,
        certification_eligible=False,
    )
    return report.to_dict()


def _sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.host_id = None
    sandbox.files = MagicMock()
    sandbox.run.return_value = SimpleNamespace(
        exit_code=0,
        stdout="",
        stderr="",
        signal=None,
        timed_out=False,
    )
    return sandbox


def _archive_member(archive: bytes, suffix: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
        matches = [
            member
            for member in bundle.getmembers()
            if member.isfile() and member.name.endswith(suffix)
        ]
        assert len(matches) == 1, f"expected exactly one archive member ending {suffix}"
        extracted = bundle.extractfile(matches[0])
        assert extracted is not None
        return extracted.read()


def _archive_names(archive: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
        return {member.name for member in bundle.getmembers()}


def test_remote_validation_uses_dedicated_sandbox_and_returns_report(
    tmp_path: Path,
) -> None:
    remote = _remote_module()
    source = tmp_path / "environment"
    write_valid_env(source)
    marker = source / "revision-marker.txt"
    marker.write_text("exact revision contents\n")
    (source / ".env").write_text("HF_TOKEN=hf_do_not_upload\n")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("credential = do-not-upload\n")
    (source / ".openenv").mkdir()
    (source / ".openenv" / "validation-report.json").write_text("{}\n")
    sandbox = _sandbox()
    uploads: dict[str, bytes] = {}

    def capture_upload(local_path: str | Path, remote_path: str, **_: object) -> None:
        uploads[remote_path] = Path(local_path).read_bytes()

    def write_download(_remote_path: str, local_path: str | Path) -> None:
        Path(local_path).write_text(json.dumps(_report_payload(source)))

    sandbox.files.upload.side_effect = capture_upload
    sandbox.files.download.side_effect = write_download

    with patch.object(Sandbox, "create", return_value=sandbox) as create:
        with patch.object(
            SandboxPool,
            "__init__",
            side_effect=AssertionError("remote validation must not create a pool"),
        ) as pool_init:
            with patch.object(
                SandboxPool,
                "create",
                side_effect=AssertionError("remote validation must not use a pool"),
            ) as pool_create:
                report = remote.run_remote_validation(
                    source,
                    profile=ValidationProfile.PUBLISH,
                    repo_sha=_REVISION,
                    sandbox_image=_SANDBOX_IMAGE,
                )

    create.assert_called_once()
    assert create.call_args.kwargs["image"] == _SANDBOX_IMAGE
    assert create.call_args.kwargs["forward_hf_token"] is False
    pool_init.assert_not_called()
    pool_create.assert_not_called()

    assert len(uploads) == 2
    source_upload = next(
        payload for path, payload in uploads.items() if "revision" in path
    )
    validator_upload = next(
        payload for path, payload in uploads.items() if "validator" in path
    )
    assert _archive_member(source_upload, "revision-marker.txt") == marker.read_bytes()
    source_members = _archive_names(source_upload)
    assert not any(name.endswith(".env") for name in source_members)
    assert not any(".git" in Path(name).parts for name in source_members)
    assert not any(name.endswith("validation-report.json") for name in source_members)
    validator_source = Path(__file__).parents[2] / "src" / "openenv" / "validation"
    assert (
        _archive_member(validator_upload, "openenv/validation/models.py")
        == (validator_source / "models.py").read_bytes()
    )

    sandbox.run.assert_called_once()
    command = sandbox.run.call_args.args[0]
    command_text = " ".join(command) if isinstance(command, list) else command
    assert "publish" in command_text
    assert "validation" in command_text
    for remote_path in uploads:
        assert remote_path in command_text
    sandbox.files.download.assert_called_once()
    downloaded_report_path = sandbox.files.download.call_args.args[0]
    assert downloaded_report_path in command_text

    assert isinstance(report, ValidationReport)
    assert report.repo_sha == _REVISION
    assert report.runner.runner == "hf-sandbox"
    assert report.runner.isolation_mode == "dedicated"
    assert report.runner.official is False
    assert report.certified is False
    assert report.certification_eligible is False
    assert report.results[0].criterion_id == "source.validation_spec"
    sandbox.close.assert_called_once_with()


def test_remote_validation_command_failure_is_clean_and_closes_sandbox(
    tmp_path: Path,
) -> None:
    remote = _remote_module()
    source = tmp_path / "environment"
    write_valid_env(source)
    sandbox = _sandbox()
    sandbox.run.return_value = SimpleNamespace(
        exit_code=17,
        stdout="",
        stderr="HF_TOKEN=hf_1234567890abcdef",
        signal=None,
        timed_out=False,
    )

    with patch.object(Sandbox, "create", return_value=sandbox):
        with pytest.raises(remote.RemoteValidationError) as exc_info:
            remote.run_remote_validation(
                source,
                profile=ValidationProfile.PUBLISH,
                repo_sha=_REVISION,
                sandbox_image=_SANDBOX_IMAGE,
            )

    message = str(exc_info.value).lower()
    assert "remote validation" in message
    assert "failed" in message
    assert "hf_1234567890abcdef" not in message
    sandbox.files.download.assert_not_called()
    sandbox.close.assert_called_once_with()


@pytest.mark.parametrize("downloaded", [b"not-json", b"{}"])
def test_remote_validation_malformed_report_is_clean_and_closes_sandbox(
    tmp_path: Path,
    downloaded: bytes,
) -> None:
    remote = _remote_module()
    source = tmp_path / "environment"
    write_valid_env(source)
    sandbox = _sandbox()

    def write_download(_remote_path: str, local_path: str | Path) -> None:
        Path(local_path).write_bytes(downloaded)

    sandbox.files.download.side_effect = write_download

    with patch.object(Sandbox, "create", return_value=sandbox):
        with pytest.raises(remote.RemoteValidationError) as exc_info:
            remote.run_remote_validation(
                source,
                profile=ValidationProfile.PUBLISH,
                repo_sha=_REVISION,
                sandbox_image=_SANDBOX_IMAGE,
            )

    message = str(exc_info.value).lower()
    assert "report" in message
    assert "invalid" in message or "malformed" in message
    assert "not-json" not in message
    sandbox.close.assert_called_once_with()
