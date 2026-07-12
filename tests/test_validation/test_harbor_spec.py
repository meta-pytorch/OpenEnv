# SPDX-License-Identifier: BSD-3-Clause

"""Tests for adapting Harbor task manifests into validation requirements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openenv.validation.specs import (
    find_harbor_verifier_script,
    load_harbor_requirements,
    NetworkMode,
    RequirementsState,
)

from ._helpers import write_harbor_task


def test_task_manifest_loader_normalizes_harbor_1_1_envelope(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        'schema_version = "1.1"\n'
        'artifacts = ["/workspace/result.json", '
        '{ source = "/workspace/logs", destination = "/logs", exclude = ["*.tmp"] }]\n'
        "\n"
        "[task]\n"
        'name = "openenv/example"\n'
        'description = "Example environment"\n'
        "\n"
        "[agent]\n"
        "timeout_sec = 120.0\n"
        "\n"
        "[verifier]\n"
        "timeout_sec = 45.0\n"
        'env = { VERIFY_TOKEN = "super-secret" }\n'
        'user = "grader"\n'
        "\n"
        "[environment]\n"
        "build_timeout_sec = 300.0\n"
        'docker_image = "ghcr.io/openenv/example@sha256:abc"\n'
        "cpus = 4\n"
        "memory_mb = 8192\n"
        "storage_mb = 20480\n"
        "gpus = 1\n"
        'gpu_types = ["A100", "H100"]\n'
        "allow_internet = false\n"
        'env = { API_TOKEN = "do-not-report" }\n'
        "\n"
        "[environment.healthcheck]\n"
        'command = "curl -f http://127.0.0.1:8000/health"\n'
        "interval_sec = 2.0\n"
        "timeout_sec = 3.0\n"
        "start_period_sec = 4.0\n"
        "start_interval_sec = 1.0\n"
        "retries = 5\n"
    )

    loaded = load_harbor_requirements(task_dir)

    assert loaded.state is RequirementsState.LOADED
    assert loaded.error is None
    assert loaded.provenance is not None
    assert loaded.provenance.source_id == "harbor"
    assert loaded.provenance.source_version == "1.1"
    assert loaded.provenance.document_digest is not None
    assert loaded.provenance.document_digest.startswith("sha256:")
    assert loaded.requirements is not None
    requirements = loaded.requirements
    assert requirements.identity.name == "openenv/example"
    assert requirements.agent.timeout_s == 120.0
    assert requirements.verifier.timeout_s == 45.0
    assert requirements.environment.cpus == 4
    assert requirements.environment.memory_mb == 8192
    assert requirements.environment.storage_mb == 20480
    assert requirements.environment.gpus == 1
    assert requirements.environment.gpu_types == ("A100", "H100")
    assert requirements.environment.network.mode is NetworkMode.DENY_ALL
    assert requirements.environment.healthcheck is not None
    assert requirements.environment.healthcheck.retries == 5
    assert requirements.artifacts[0].source == "/workspace/result.json"
    assert requirements.artifacts[1].destination == "/logs"
    assert requirements.artifacts[1].exclude == ("*.tmp",)

    evidence = json.dumps(loaded.to_evidence(), sort_keys=True)
    assert "API_TOKEN" in evidence
    assert "VERIFY_TOKEN" in evidence
    assert "do-not-report" not in evidence
    assert "super-secret" not in evidence


def test_task_manifest_loader_distinguishes_absent_and_invalid(
    tmp_path: Path,
) -> None:
    absent = load_harbor_requirements(tmp_path)
    assert absent.state is RequirementsState.ABSENT
    assert absent.requirements is None

    (tmp_path / "task.toml").write_text("[environment\ncpus = 2\n")
    invalid = load_harbor_requirements(tmp_path)
    assert invalid.state is RequirementsState.INVALID
    assert invalid.requirements is None
    assert invalid.error


def test_task_manifest_legacy_version_alias_is_supported(tmp_path: Path) -> None:
    (tmp_path / "task.toml").write_text(
        'version = "1.1"\n[environment]\nallow_internet = true\n'
    )

    loaded = load_harbor_requirements(tmp_path)

    assert loaded.state is RequirementsState.LOADED
    assert loaded.requirements is not None
    assert loaded.provenance is not None
    assert loaded.provenance.source_version == "1.1"
    assert loaded.requirements.environment.network.mode is NetworkMode.ALLOW_ALL


def test_task_manifest_unknown_fields_are_structured_invalid(tmp_path: Path) -> None:
    (tmp_path / "task.toml").write_text(
        'schema_version = "1.1"\n[environment]\ncpus = 2\ninvented = true\n'
    )

    loaded = load_harbor_requirements(tmp_path)

    assert loaded.state is RequirementsState.INVALID
    assert loaded.error is not None
    assert "invented" in loaded.error


def test_task_manifest_reports_unsupported_schema(tmp_path: Path) -> None:
    (tmp_path / "task.toml").write_text('schema_version = "2.0"\n')

    loaded = load_harbor_requirements(tmp_path)

    assert loaded.state is RequirementsState.UNSUPPORTED
    assert loaded.provenance is not None
    assert loaded.provenance.source_version == "2.0"
    assert loaded.error is not None


@pytest.mark.parametrize(
    "body, expected",
    [
        ('artifacts = ["relative/result.json"]\n', "absolute container paths"),
        (
            '[environment]\ngpus = 0\ngpu_types = ["A100"]\n',
            "positive gpus",
        ),
    ],
)
def test_harbor_specific_invariants_stay_in_adapter(
    tmp_path: Path, body: str, expected: str
) -> None:
    (tmp_path / "task.toml").write_text(f'schema_version = "1.1"\n{body}')

    loaded = load_harbor_requirements(tmp_path)

    assert loaded.state is RequirementsState.INVALID
    assert loaded.error is not None
    assert expected in loaded.error


@pytest.mark.parametrize(
    "memory",
    ["1e999G", "sNaNG", "1e9999999999999999G"],
)
def test_task_manifest_rejects_nonfinite_legacy_resources(
    tmp_path: Path, memory: str
) -> None:
    (tmp_path / "task.toml").write_text(
        f'schema_version = "1.1"\n[environment]\nmemory = "{memory}"\n'
    )

    loaded = load_harbor_requirements(tmp_path)

    assert loaded.state is RequirementsState.INVALID
    assert loaded.error is not None


def test_task_manifest_rejects_excessive_nesting(tmp_path: Path) -> None:
    nested = "0"
    for _ in range(500):
        nested = f"[{nested}]"
    (tmp_path / "task.toml").write_text(
        f'schema_version = "1.1"\nmetadata = {{ nested = {nested} }}\n'
    )

    loaded = load_harbor_requirements(tmp_path)

    assert loaded.state is RequirementsState.INVALID
    assert loaded.error is not None


def test_task_manifest_rejects_symbolic_link(tmp_path: Path) -> None:
    source = tmp_path / "submitted.toml"
    source.write_text('schema_version = "1.1"\n')
    try:
        (tmp_path / "task.toml").symlink_to(source)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    loaded = load_harbor_requirements(tmp_path)

    assert loaded.state is RequirementsState.INVALID
    assert loaded.error is not None
    assert "symbolic link" in loaded.error


def test_task_manifest_rejects_dangling_link_and_directory(tmp_path: Path) -> None:
    try:
        (tmp_path / "task.toml").symlink_to(tmp_path / "missing.toml")
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    dangling = load_harbor_requirements(tmp_path)
    assert dangling.state is RequirementsState.INVALID

    (tmp_path / "task.toml").unlink()
    (tmp_path / "task.toml").mkdir()
    directory = load_harbor_requirements(tmp_path)
    assert directory.state is RequirementsState.INVALID


def test_verifier_discovery_rejects_symlinked_tests_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test.sh").write_text("#!/bin/sh\n")
    environment = tmp_path / "environment"
    environment.mkdir()
    (tmp_path / "task.toml").write_text('schema_version = "1.1"\n')
    try:
        (tmp_path / "tests").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    assert find_harbor_verifier_script(environment) is None


def test_harbor_step_reward_map_and_sibling_verifier_are_loaded(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "harbor-task"
    environment = write_harbor_task(task_root)

    loaded = load_harbor_requirements(environment)

    assert loaded.state is RequirementsState.LOADED
    assert loaded.requirements is not None
    assert loaded.requirements.steps[0].min_reward == {
        "correctness": 0.8,
        "style": 0.5,
    }
    assert loaded.requirements.steps[0].artifacts[0].source == (
        "/workspace/result.json"
    )
    assert find_harbor_verifier_script(environment) == (task_root / "tests" / "test.sh")
