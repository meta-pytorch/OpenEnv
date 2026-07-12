# SPDX-License-Identifier: BSD-3-Clause

"""Tests for typed, display-only validation guidance."""

from __future__ import annotations

import pytest
from openenv.validation import (
    DiagnosticLocation,
    RunnerCapabilities,
    ValidationDiagnostic,
    ValidationProfile,
    ValidationRemediation,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


def test_guidance_serializes_as_typed_report_data() -> None:
    result = ValidationResult(
        criterion_id="source.lockfile",
        requirement="A lockfile is required",
        status=ValidationStatus.FAIL,
        severity=ValidationSeverity.BLOCKING,
        evidence={"missing": True},
        duration_s=0.01,
        timeout_s=1.0,
        required_capabilities=frozenset(),
        diagnostics=(
            ValidationDiagnostic(
                code="lockfile_missing",
                message="Generate a lockfile",
                location=DiagnosticLocation(path="uv.lock", line=1),
            ),
        ),
        remediation=(
            ValidationRemediation(
                kind="command",
                message="Generate the lockfile",
                argv=("uv", "lock"),
                cwd=".",
            ),
        ),
    )
    report = ValidationReport(
        target=".",
        profile=ValidationProfile.PUBLISH,
        policy_version="test-policy",
        runner=RunnerCapabilities(runner="local"),
        results=(result,),
        duration_s=0.01,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:00.010000+00:00",
    )

    criterion = report.to_dict()["criteria"][0]

    assert criterion["diagnostics"] == [
        {
            "code": "lockfile_missing",
            "message": "Generate a lockfile",
            "location": {
                "path": "uv.lock",
                "pointer": None,
                "line": 1,
                "column": None,
            },
        }
    ]
    assert criterion["remediation"] == [
        {
            "kind": "command",
            "message": "Generate the lockfile",
            "argv": ["uv", "lock"],
            "cwd": ".",
            "path": None,
            "pointer": None,
            "url": None,
        }
    ]


@pytest.mark.parametrize("path", ["/tmp/secret", "../secret", "dir/../../secret"])
def test_guidance_locations_must_be_repository_relative(path: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        DiagnosticLocation(path=path)


def test_command_guidance_requires_argv_and_never_accepts_a_shell_string() -> None:
    with pytest.raises(ValueError, match="requires argv"):
        ValidationRemediation(kind="command", message="Run the fix")

    with pytest.raises(TypeError):
        ValidationRemediation(  # type: ignore[call-arg]
            kind="command",
            message="Run the fix",
            command="uv lock",
        )


def test_documentation_guidance_requires_a_credential_free_https_url() -> None:
    with pytest.raises(ValueError, match="HTTPS URL"):
        ValidationRemediation(
            kind="documentation",
            message="Read the guide",
            url="http://user:password@example.com/guide",
        )
