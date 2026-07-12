# SPDX-License-Identifier: BSD-3-Clause

"""Tests for local validation profiles and runtime launch helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from openenv.validation.local import (
    _runtime_environment,
    _server_command,
    _subject_process_kwargs,
    run_local_validation,
)
from openenv.validation.models import (
    CheckOutcome,
    ValidationCapability,
    ValidationCheck,
    ValidationPolicy,
    ValidationProfile,
    ValidationSeverity,
)
from openenv.validation.specs import (
    AdapterIdentity,
    DetectionMode,
    ExecutionModel,
    RequirementsLoad,
    RequirementsState,
    SpecIdentity,
    SpecLoad,
    SpecLoadState,
    ValidationRequirements,
    ValidationSpecRegistry,
    ValidationSubject,
)

from ._helpers import write_valid_env


class _OneShotAdapter:
    spec_id = "one-shot-local"
    adapter_id = "one-shot-local-adapter"
    adapter_version = "1"
    execution_model = ExecutionModel.ONE_SHOT
    signature_files = ("task.oneshot",)

    def detect(self, root: Path) -> bool:
        return (root / "task.oneshot").is_file()

    def inspect(self, root: Path) -> SpecLoad:
        identity = SpecIdentity(
            spec_id=self.spec_id,
            spec_version="1",
            adapter=AdapterIdentity(self.adapter_id, self.adapter_version),
            execution_model=self.execution_model,
        )
        if not self.detect(root):
            return SpecLoad(state=SpecLoadState.ABSENT, identity=identity)
        return SpecLoad(
            state=SpecLoadState.LOADED,
            subject=ValidationSubject(
                spec=identity,
                signature_path="task.oneshot",
                detection_mode=DetectionMode.AUTO,
                requirements=RequirementsLoad(
                    state=RequirementsState.LOADED,
                    requirements=ValidationRequirements(),
                ),
            ),
        )


def test_static_profile_skips_absent_requirements_without_certifying(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)

    report = run_local_validation(env_dir, profile=ValidationProfile.STATIC)
    payload = report.to_dict()

    requirements_result = next(
        result
        for result in payload["criteria"]
        if result["id"] == "source.validation_requirements"
    )
    assert requirements_result["status"] == "skip"
    assert payload["passed"] is True
    assert payload["certified"] is False
    assert payload["certification_eligible"] is False
    assert payload["policy_version"]
    assert payload["spec"]["id"] == "openenv"
    assert payload["spec"]["execution_model"] == "served"
    assert "repo_sha" in payload


def test_present_unsupported_requirements_are_blocking(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    (env_dir / "task.toml").write_text('schema_version = "2.0"\n')

    report = run_local_validation(env_dir, profile=ValidationProfile.STATIC)
    result = next(
        result
        for result in report.results
        if result.criterion_id == "source.validation_requirements"
    )

    assert result.status.value == "fail"
    assert result.severity is ValidationSeverity.BLOCKING
    assert report.passed is False


def test_unsupported_spec_retains_top_level_provenance(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    (env_dir / "openenv.yaml").write_text("spec_version: 2\n")

    report = run_local_validation(env_dir, profile=ValidationProfile.STATIC)
    payload = report.to_dict()

    assert report.spec is None
    assert report.spec_identity is not None
    assert payload["spec"] == {
        "adapter": {"id": "openenv-yaml", "version": "1"},
        "execution_model": "served",
        "id": "openenv",
        "version": "2",
    }


def test_one_shot_spec_is_not_launched_as_an_openenv_server(tmp_path: Path) -> None:
    (tmp_path / "task.oneshot").write_text("signature")
    runtime_check = ValidationCheck(
        criterion_id="spec.one_shot.replay",
        requirement="Replay the one-shot package",
        capabilities=frozenset({ValidationCapability.RUNTIME}),
        severity=ValidationSeverity.BLOCKING,
        timeout_s=1.0,
        evaluator=lambda _context: CheckOutcome.pass_(),
    )
    policy = ValidationPolicy(
        version="one-shot-local-v1",
        supported_subjects=frozenset({("one-shot-local", ExecutionModel.ONE_SHOT)}),
        checks=(runtime_check,),
    )

    with patch("openenv.validation.local._launch_environment") as launch:
        report = run_local_validation(
            tmp_path,
            profile=ValidationProfile.RUNTIME,
            spec_registry=ValidationSpecRegistry((_OneShotAdapter(),)),
            policy=policy,
        )

    launch.assert_not_called()
    assert report.spec is not None
    assert report.spec.spec.execution_model is ExecutionModel.ONE_SHOT
    assert report.results[0].status.value == "skip"
    assert report.results[0].evidence["unavailable_reasons"] == {
        "runtime": "unsupported_execution_model:one_shot"
    }


def test_static_validation_never_reads_symlinked_source_files(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    env_dir.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text(
        "spec_version: 1\n"
        "name: TOPSECRET_VALUE\n"
        "runtime: fastapi\n"
        "app: server.app:app\n"
        "port: 8000\n"
    )
    try:
        (env_dir / "openenv.yaml").symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    report = run_local_validation(env_dir, profile=ValidationProfile.STATIC)
    serialized = json.dumps(report.to_dict())

    assert report.passed is False
    assert "TOPSECRET_VALUE" not in serialized
    openenv_manifest = next(
        result
        for result in report.results
        if result.criterion_id == "source.openenv_manifest"
    )
    assert openenv_manifest.status.value == "fail"


def test_static_checks_reject_symlinked_project_files(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    write_valid_env(env_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    replacements = {
        env_dir / "pyproject.toml": outside / "pyproject.toml",
        env_dir / "uv.lock": outside / "uv.lock",
        env_dir / "server" / "app.py": outside / "app.py",
        env_dir / "server" / "Dockerfile": outside / "Dockerfile",
    }
    for target, source in replacements.items():
        source.write_text(target.read_text())
        target.unlink()
        try:
            target.symlink_to(source)
        except OSError:
            pytest.skip("symbolic links are unavailable on this platform")

    report = run_local_validation(env_dir, profile=ValidationProfile.STATIC)
    by_id = {result.criterion_id: result for result in report.results}

    assert by_id["source.project_layout"].status.value == "fail"
    assert by_id["source.dependencies"].status.value == "fail"
    assert by_id["source.lockfile"].status.value == "fail"
    assert by_id["source.schema_sources"].status.value == "fail"
    assert by_id["source.dockerfile"].status.value == "fail"


def test_local_runtime_does_not_forward_ambient_credentials(tmp_path: Path) -> None:
    ambient = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "HF_TOKEN": "hf-secret",
        "OPENAI_API_KEY": "openai-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "PYTHONPATH": "extra-python-path",
    }

    with patch.dict(os.environ, ambient, clear=True):
        process_env = _runtime_environment(tmp_path)

    assert process_env["HOME"] == str(tmp_path)
    assert process_env["OPENENV_VALIDATION"] == "1"
    assert "extra-python-path" in process_env["PYTHONPATH"]
    assert "HF_TOKEN" not in process_env
    assert "OPENAI_API_KEY" not in process_env
    assert "AWS_SECRET_ACCESS_KEY" not in process_env


def test_server_command_honors_manifest_port(tmp_path: Path) -> None:
    with patch("openenv.validation.local.shutil.which", return_value=None):
        command = _server_command(tmp_path, "server.app:app", 8123)

    assert command[1:4] == ["-m", "uvicorn", "server.app:app"]
    assert command[-3:] == ["8123", "--log-level", "warning"]


def test_static_profile_rejects_url_in_public_api() -> None:
    with pytest.raises(ValueError, match="local source"):
        run_local_validation("https://example.com", profile=ValidationProfile.STATIC)


def test_remote_subject_process_drops_to_declared_unprivileged_identity() -> None:
    with patch.dict(
        os.environ,
        {
            "OPENENV_VALIDATION_SUBJECT_UID": "10001",
            "OPENENV_VALIDATION_SUBJECT_GID": "10002",
        },
        clear=False,
    ):
        assert _subject_process_kwargs() == {
            "user": 10001,
            "group": 10002,
            "extra_groups": (),
        }
