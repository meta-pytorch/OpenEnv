# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the openenv validate command and runtime validation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from openenv.cli.__main__ import app
from openenv.cli._validation import validate_running_environment
from openenv.validation import ValidationProfile
from typer.testing import CliRunner


runner = CliRunner()


class _MockResponse:
    """Minimal mock response object for requests.get/post tests."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


def _write_minimal_valid_env(
    env_dir: Path,
    *,
    main_signature: str = "def main():",
    main_invocation: str = "main()",
) -> None:
    """Create a minimal local environment that passes local validation."""
    (env_dir / "server").mkdir(parents=True)

    (env_dir / "openenv.yaml").write_text(
        "spec_version: 1\nname: test_env\ntype: space\nruntime: fastapi\napp: server.app:app\nport: 8000\n"
    )
    (env_dir / "uv.lock").write_text(
        'version = 1\nrevision = 1\nrequires-python = ">=3.10"\n'
    )
    (env_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-env"\n'
        'version = "0.1.0"\n'
        'dependencies = ["openenv>=0.2.0"]\n'
        "\n"
        "[project.scripts]\n"
        'server = "server.app:main"\n'
    )
    (env_dir / "server" / "app.py").write_text(
        f"{main_signature}\n    return None\n\nif __name__ == '__main__':\n    {main_invocation}\n"
    )
    (env_dir / "server" / "Dockerfile").write_text("FROM python:3.12-slim\n")


def _diagnostic_failure_report(target: Path):
    from openenv.validation.models import (
        DiagnosticLocation,
        RunnerCapabilities,
        ValidationCapability,
        ValidationDiagnostic,
        ValidationProfile,
        ValidationRemediation,
        ValidationReport,
        ValidationResult,
        ValidationSeverity,
        ValidationStatus,
    )

    diagnostic = ValidationDiagnostic(
        code="missing_dependency",
        message="Missing OpenEnv runtime dependency",
        location=DiagnosticLocation(path="pyproject.toml"),
    )
    remediation = ValidationRemediation(
        kind="command",
        message="Add the OpenEnv runtime dependency",
        argv=("uv", "add", "openenv>=0.2.0"),
        cwd=".",
    )
    result = ValidationResult(
        criterion_id="source.dependencies",
        requirement="The environment declares the OpenEnv runtime dependency",
        status=ValidationStatus.FAIL,
        severity=ValidationSeverity.BLOCKING,
        evidence={
            "project_dependency_count": 0,
            "dockerfile_installs_openenv": False,
        },
        duration_s=0.01,
        timeout_s=10.0,
        required_capabilities=frozenset({ValidationCapability.SOURCE}),
        diagnostics=(diagnostic,),
        remediation=(remediation,),
    )
    return ValidationReport(
        target=str(target),
        profile=ValidationProfile.STATIC,
        policy_version="test-policy",
        runner=RunnerCapabilities(
            runner="local",
            available=frozenset({ValidationCapability.SOURCE}),
        ),
        results=(result,),
        duration_s=0.01,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:00.010000+00:00",
    )


def test_validate_running_environment_success() -> None:
    """Runtime validator returns passing criteria for a conforming server."""

    def _fake_get(url: str, timeout: float, *, allow_redirects: bool) -> _MockResponse:
        assert allow_redirects is False
        if url.endswith("/openapi.json"):
            return _MockResponse(
                200,
                {
                    "info": {"version": "1.0.0"},
                    "paths": {
                        "/health": {},
                        "/metadata": {},
                        "/schema": {},
                        "/mcp": {},
                        "/reset": {},
                        "/step": {},
                        "/state": {},
                    },
                },
            )
        if url.endswith("/health"):
            return _MockResponse(200, {"status": "healthy"})
        if url.endswith("/metadata"):
            return _MockResponse(200, {"name": "EchoEnv", "description": "Echo env"})
        if url.endswith("/schema"):
            return _MockResponse(
                200,
                {"action": {"type": "object"}, "observation": {}, "state": {}},
            )
        raise AssertionError(f"Unexpected GET url: {url}")

    def _fake_post(
        url: str, json: dict, timeout: float, *, allow_redirects: bool
    ) -> _MockResponse:
        assert allow_redirects is False
        if url.endswith("/mcp"):
            return _MockResponse(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                },
            )
        raise AssertionError(f"Unexpected POST url: {url}")

    with patch("openenv.cli._validation.requests.get", side_effect=_fake_get):
        with patch("openenv.cli._validation.requests.post", side_effect=_fake_post):
            report = validate_running_environment("http://localhost:8000")

    assert report["passed"] is True
    assert report["standard_version"] == "1.0.0"
    assert report["mode"] == "simulation"
    assert report["validation_type"] == "running_environment"
    assert report["summary"]["passed_count"] == 6
    assert report["summary"]["total_count"] == 6
    assert report["summary"]["failed_criteria"] == []


def test_validate_running_environment_failure() -> None:
    """Runtime validator marks report as failed when criteria fail."""

    def _fake_get(url: str, timeout: float, *, allow_redirects: bool) -> _MockResponse:
        assert allow_redirects is False
        if url.endswith("/openapi.json"):
            return _MockResponse(
                200,
                {
                    "info": {"version": "1.0.0"},
                    "paths": {
                        "/health": {},
                        "/metadata": {},
                        "/schema": {},
                        "/mcp": {},
                    },
                },
            )
        if url.endswith("/health"):
            return _MockResponse(200, {"status": "healthy"})
        if url.endswith("/metadata"):
            return _MockResponse(500, {"detail": "boom"})
        if url.endswith("/schema"):
            return _MockResponse(
                200,
                {"action": {"type": "object"}, "observation": {}, "state": {}},
            )
        raise AssertionError(f"Unexpected GET url: {url}")

    def _fake_post(
        url: str, json: dict, timeout: float, *, allow_redirects: bool
    ) -> _MockResponse:
        assert allow_redirects is False
        if url.endswith("/mcp"):
            return _MockResponse(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                },
            )
        raise AssertionError(f"Unexpected POST url: {url}")

    with patch("openenv.cli._validation.requests.get", side_effect=_fake_get):
        with patch("openenv.cli._validation.requests.post", side_effect=_fake_post):
            report = validate_running_environment("http://localhost:8000")

    assert report["passed"] is False
    metadata_checks = [c for c in report["criteria"] if c["id"] == "metadata_endpoint"]
    assert metadata_checks
    assert metadata_checks[0]["passed"] is False
    assert report["summary"]["passed_count"] == 5
    assert report["summary"]["total_count"] == 6
    assert report["summary"]["failed_criteria"] == ["metadata_endpoint"]


def test_validate_command_runtime_target_outputs_json() -> None:
    """CLI validates runtime targets through the shared report path."""
    payload = {
        "target": "https://example.com",
        "validation_type": "openenv_validation",
        "report_schema_version": "1.0",
        "profile": "runtime",
        "passed": True,
        "criteria": [],
    }
    mock_report = MagicMock()
    mock_report.passed = True
    mock_report.to_dict.return_value = payload

    with patch(
        "openenv.cli.commands.validate.run_local_validation",
        return_value=mock_report,
    ) as mock_validate:
        result = runner.invoke(app, ["validate", "https://example.com"])

    assert result.exit_code == 0
    assert json.loads(result.output) == payload
    mock_validate.assert_called_once_with(
        "https://example.com",
        profile=ValidationProfile.RUNTIME,
        runtime_url="https://example.com",
        timeout_s=5.0,
    )


def test_validate_shared_profile_reports_invalid_runtime_url() -> None:
    result = runner.invoke(
        app,
        ["validate", "--url", "http://[::1", "--profile", "runtime", "--json"],
    )

    assert result.exit_code == 1
    assert "Error: Invalid runtime URL" in result.output


def test_validate_command_local_path_still_works(tmp_path: Path) -> None:
    """CLI local validation remains backward compatible."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code == 0
    assert "[OK]" in result.output


def test_unqualified_local_validate_uses_shared_static_semantics(
    tmp_path: Path,
) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "openenv.yaml").write_text("spec_version: [\n")

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code == 1
    assert "static validation" in result.output
    assert "source.validation_spec" in result.output
    assert "The validation spec could not be loaded" in result.output


def test_shared_human_report_surfaces_typed_remediation(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    report = _diagnostic_failure_report(env_dir)

    with patch(
        "openenv.cli.commands.validate.run_local_validation",
        return_value=report,
    ):
        result = runner.invoke(
            app,
            ["validate", str(env_dir), "--profile", "static"],
        )

    assert result.exit_code == 1
    assert "Missing OpenEnv runtime dependency" in result.output
    assert "pyproject.toml" in result.output
    assert "Add the OpenEnv runtime dependency" in result.output
    assert "uv add 'openenv>=0.2.0'" in result.output
    assert "project_dependency_count" not in result.output


def test_shared_verbose_report_includes_structured_evidence(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    report = _diagnostic_failure_report(env_dir)

    with patch(
        "openenv.cli.commands.validate.run_local_validation",
        return_value=report,
    ):
        result = runner.invoke(
            app,
            ["validate", str(env_dir), "--profile", "static", "--verbose"],
        )

    assert result.exit_code == 1
    assert "project_dependency_count" in result.output
    assert "dockerfile_installs_openenv" in result.output


def test_validate_remote_defaults_to_publish_profile(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    remote_report = MagicMock()
    remote_report.passed = True
    remote_report.to_dict.return_value = {
        "validation_type": "openenv_validation",
        "report_schema_version": "1.0",
        "profile": "publish",
        "passed": True,
        "criteria": [],
    }

    with (
        patch(
            "openenv.cli.commands.validate.run_remote_validation",
            return_value=remote_report,
            create=True,
        ) as run_remote,
        patch("openenv.cli.commands.validate.run_local_validation") as run_local,
    ):
        result = runner.invoke(
            app,
            ["validate", str(env_dir), "--remote", "--json"],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["profile"] == "publish"
    run_remote.assert_called_once_with(
        env_dir,
        profile=ValidationProfile.PUBLISH,
        flavor="cpu-basic",
        runtime_timeout_s=5.0,
    )
    run_local.assert_not_called()


def test_validate_remote_rejects_runtime_url() -> None:
    result = runner.invoke(
        app,
        ["validate", "https://example.com", "--remote"],
    )

    assert result.exit_code == 1
    assert "--remote requires a local source directory" in result.output


def test_validate_command_local_json_output(tmp_path: Path) -> None:
    """CLI can emit JSON report for local validation via --json."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)

    result = runner.invoke(app, ["validate", str(env_dir), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["validation_type"] == "openenv_validation"
    assert payload["report_schema_version"] == "1.0"
    assert payload["profile"] == "static"
    assert payload["spec"]["id"] == "openenv"
    assert payload["spec"]["adapter"] == {
        "id": "openenv-yaml",
        "version": "1",
    }
    assert payload["spec"]["execution_model"] == "served"
    assert payload["passed"] is True
    assert payload["summary"]["passed_count"] >= 1
    assert payload["summary"]["total_count"] >= 1
    assert payload["summary"]["failed_criteria"] == []


def test_profile_json_can_be_written_to_output_file(tmp_path: Path) -> None:
    env_dir = tmp_path / "test_env"
    report_path = tmp_path / "reports" / "validation.json"
    _write_minimal_valid_env(env_dir)

    result = runner.invoke(
        app,
        [
            "validate",
            str(env_dir),
            "--profile",
            "static",
            "--json",
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    stdout_payload = json.loads(result.output)
    file_payload = json.loads(report_path.read_text())
    assert file_payload == stdout_payload
    assert file_payload["profile"] == "static"
    assert file_payload["report_schema_version"] == "1.0"


def test_validate_command_rejects_environment_package_as_runtime_dependency(
    tmp_path: Path,
) -> None:
    """An openenv-* environment package is not the OpenEnv runtime package."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-env"\n'
        'version = "0.1.0"\n'
        'dependencies = ["openenv-echo-env>=0.1.0"]\n'
        "\n"
        "[project.scripts]\n"
        'server = "server.app:main"\n'
    )

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code != 0
    assert "Missing required dependency: openenv>=0.2.0" in result.output


def test_validate_command_accepts_dockerfile_managed_openenv_runtime(
    tmp_path: Path,
) -> None:
    """Local validation accepts envs that install OpenEnv in Dockerfile."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-env"\n'
        'version = "0.1.0"\n'
        'dependencies = ["fastapi>=0.115.0"]\n'
        "\n"
        "[project.scripts]\n"
        'server = "server.app:main"\n'
    )
    (env_dir / "server" / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        'RUN pip install --no-cache-dir --no-deps "openenv[core]>=0.2.2"\n'
    )

    result = runner.invoke(app, ["validate", str(env_dir), "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["passed"] is True


def test_validate_command_accepts_main_call_with_arguments(tmp_path: Path) -> None:
    """Local validation accepts a guarded main(...) call with arguments."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(
        env_dir,
        main_signature="def main(port: int = 8000):",
        main_invocation="main(port=8000)",
    )

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code == 0
    assert "main() function not callable" not in result.output
    assert "[OK]" in result.output


def test_validate_command_rejects_nested_main_guard(tmp_path: Path) -> None:
    """Local validation requires the __main__ guard at module scope."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "server" / "app.py").write_text(
        "def main():\n    return None\n\n"
        "def wrapper():\n"
        "    if __name__ == '__main__':\n"
        "        main()\n"
    )

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code != 0
    assert "main() function not callable" in result.output


def test_validate_command_accepts_later_top_level_main_guard(tmp_path: Path) -> None:
    """Local validation scans each top-level __main__ guard for a main() call."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "server" / "app.py").write_text(
        "def main():\n    return None\n\n"
        "if __name__ == '__main__':\n"
        "    print('starting')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code == 0
    assert "[OK]" in result.output


def test_validate_command_syntax_error_fallback_requires_dunder_main(
    tmp_path: Path,
) -> None:
    """Syntax-error fallback still requires the literal __main__ guard string."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)
    (env_dir / "server" / "app.py").write_text(
        "def main(:\n    return None\n\nif __name__ ==\n    main(\n"
    )

    result = runner.invoke(app, ["validate", str(env_dir)])

    assert result.exit_code != 0
    assert "main() function not callable" in result.output


def test_validate_command_rejects_mixed_path_and_url(tmp_path: Path) -> None:
    """CLI rejects mixing a local path argument with --url mode."""
    env_dir = tmp_path / "test_env"
    _write_minimal_valid_env(env_dir)

    result = runner.invoke(
        app,
        ["validate", str(env_dir), "--url", "http://localhost:8000"],
    )

    assert result.exit_code != 0
    assert "Cannot combine a local path argument with --url" in result.output
