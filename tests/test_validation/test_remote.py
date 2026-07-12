# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for author-triggered validation in a dedicated HF Sandbox."""

from __future__ import annotations

import importlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import Sandbox, SandboxPool
from openenv.validation import build_validation_plan, execute_validation_plan
from openenv.validation.models import (
    RunnerCapabilities,
    ValidationCapability,
    ValidationProfile,
    ValidationReport,
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
    capabilities = RunnerCapabilities(
        runner="hf-sandbox",
        available=frozenset(
            {ValidationCapability.SOURCE, ValidationCapability.RUNTIME}
        ),
        official=False,
        isolation_mode="dedicated",
    )
    plan, context = build_validation_plan(
        target,
        profile=ValidationProfile.PUBLISH,
        capabilities=capabilities,
    )
    report = execute_validation_plan(plan, context)
    report = replace(
        report,
        runner=capabilities,
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
    (source / ".envrc").write_text("export TOKEN=do-not-upload\n")
    (source / ".netrc").write_text("password do-not-upload\n")
    (source / "credentials.json").write_text('{"token": "do-not-upload"}\n')
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
    assert not any(name.endswith(".envrc") for name in source_members)
    assert not any(name.endswith(".netrc") for name in source_members)
    assert not any(name.endswith("credentials.json") for name in source_members)
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
    assert "setpriv" in command_text
    assert "/workspace/trusted/validation-report.json" in command_text
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
    assert report.source_digest is not None
    assert report.source_digest.startswith("sha256:")
    sandbox.kill.assert_called_once_with()
    sandbox.close.assert_called_once_with()


def test_remote_validation_preserves_parent_harbor_composition(
    tmp_path: Path,
) -> None:
    remote = _remote_module()
    task_root = tmp_path / "task"
    source = task_root / "environment"
    write_valid_env(source)
    (task_root / "task.toml").write_text('schema_version = "1.1"\n')
    (task_root / "tests").mkdir()
    (task_root / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    sandbox = _sandbox()
    uploads: dict[str, bytes] = {}

    def capture_upload(local_path: str | Path, remote_path: str, **_: object) -> None:
        uploads[remote_path] = Path(local_path).read_bytes()

    def write_download(_remote_path: str, local_path: str | Path) -> None:
        Path(local_path).write_text(json.dumps(_report_payload(source)))

    sandbox.files.upload.side_effect = capture_upload
    sandbox.files.download.side_effect = write_download
    with patch.object(Sandbox, "create", return_value=sandbox):
        report = remote.run_remote_validation(
            source,
            profile=ValidationProfile.PUBLISH,
            repo_sha=_REVISION,
            sandbox_image=_SANDBOX_IMAGE,
        )

    revision = next(payload for path, payload in uploads.items() if "revision" in path)
    members = _archive_names(revision)
    assert "task.toml" in members
    assert "tests/test.sh" in members
    command = sandbox.run.call_args.args[0]
    assert command[-1] == "environment"
    assert report.spec is not None
    assert report.spec.requirements.state.value == "loaded"


def test_remote_validation_rejects_source_symlinks(tmp_path: Path) -> None:
    remote = _remote_module()
    source = tmp_path / "environment"
    write_valid_env(source)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    (source / "linked-secret.txt").symlink_to(outside)

    with patch.object(Sandbox, "create") as create:
        with pytest.raises(remote.RemoteValidationError, match="symbolic links"):
            remote.run_remote_validation(source)

    create.assert_not_called()


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
    sandbox.kill.assert_called_once_with()
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
    sandbox.kill.assert_called_once_with()
    sandbox.close.assert_called_once_with()


@pytest.mark.parametrize("malformation", ["empty_criteria", "wrong_policy"])
def test_remote_validation_rejects_structurally_forged_report(
    tmp_path: Path,
    malformation: str,
) -> None:
    remote = _remote_module()
    source = tmp_path / "environment"
    write_valid_env(source)
    sandbox = _sandbox()
    payload = _report_payload(source)
    if malformation == "empty_criteria":
        payload["criteria"] = []
        payload["passed"] = True
        payload["status"] = "pass"
    else:
        payload["policy_version"] = "attacker-policy"

    def write_download(_remote_path: str, local_path: str | Path) -> None:
        Path(local_path).write_text(json.dumps(payload))

    sandbox.files.download.side_effect = write_download
    with patch.object(Sandbox, "create", return_value=sandbox):
        with pytest.raises(remote.RemoteValidationError, match="report"):
            remote.run_remote_validation(
                source,
                profile=ValidationProfile.PUBLISH,
                repo_sha=_REVISION,
                sandbox_image=_SANDBOX_IMAGE,
            )

    sandbox.kill.assert_called_once_with()
    sandbox.close.assert_called_once_with()
