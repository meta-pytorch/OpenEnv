# SPDX-License-Identifier: BSD-3-Clause

"""
Validation utilities for multi-mode deployment readiness.

This module provides functions to check if environments are properly
configured for multi-mode deployment (Docker, direct Python, notebooks, clusters).
"""

import re
from pathlib import Path
from typing import Any

import requests as requests
from openenv.validation.runtime_probe import (
    _build_summary as _build_summary,
    _make_criterion as _make_criterion,
    _normalize_runtime_url as _normalize_runtime_url,
    _runtime_standard_profile as _runtime_standard_profile,
    validate_running_environment as validate_running_environment,
)
from openenv.validation.source_inspection import (
    _contains_main_call as _contains_main_call,
    _dockerfile_installs_openenv_runtime as _dockerfile_installs_openenv_runtime,
    _has_main_guard_call as _has_main_guard_call,
    _is_main_guard as _is_main_guard,
    _OPENENV_DOCKER_INSTALL_RE as _OPENENV_DOCKER_INSTALL_RE,
)

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


_OPENENV_RUNTIME_DEP_RE = re.compile(r"^openenv(?:\s*(?:$|[<>=!~@;])|\[)")
_LEGACY_OPENENV_CORE_DEP_RE = re.compile(r"^openenv-core(?:\s*(?:$|[<>=!~@;])|\[)")


def validate_multi_mode_deployment(env_path: Path) -> tuple[bool, list[str]]:
    """
    Validate that an environment is ready for multi-mode deployment.

    Checks:
    1. pyproject.toml exists
    2. uv.lock exists
    3. pyproject.toml has [project.scripts] with server entry point
    4. server/app.py has a main() function
    5. Required dependencies are present

    Returns:
        `tuple` of `(is_valid, issues)` where `is_valid` is a `bool` and `issues` is a
        `list` of issue strings found during validation.
    """
    issues = []

    # Check pyproject.toml exists
    pyproject_path = env_path / "pyproject.toml"
    if not pyproject_path.exists():
        issues.append("Missing pyproject.toml")
        return False, issues

    # Check uv.lock exists
    lockfile_path = env_path / "uv.lock"
    if not lockfile_path.exists():
        issues.append("Missing uv.lock - run 'uv lock' to generate it")

    # Parse pyproject.toml
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
    except Exception as e:
        issues.append(f"Failed to parse pyproject.toml: {e}")
        return False, issues

    # Check [project.scripts] section
    scripts = pyproject.get("project", {}).get("scripts", {})
    if "server" not in scripts:
        issues.append("Missing [project.scripts] server entry point")

    # Check server entry point format
    server_entry = scripts.get("server", "")
    if server_entry and ":main" not in server_entry:
        issues.append(
            f"Server entry point should reference main function, got: {server_entry}"
        )

    # Check required dependencies
    deps = [dep.lower() for dep in pyproject.get("project", {}).get("dependencies", [])]
    has_openenv = any(_OPENENV_RUNTIME_DEP_RE.match(dep) for dep in deps)
    has_legacy_core = any(_LEGACY_OPENENV_CORE_DEP_RE.match(dep) for dep in deps)
    has_dockerfile_core = _dockerfile_installs_openenv_runtime(env_path)

    if not (has_openenv or has_legacy_core or has_dockerfile_core):
        issues.append("Missing required dependency: openenv>=0.2.0")

    # Check server/app.py exists
    server_app = env_path / "server" / "app.py"
    if not server_app.exists():
        issues.append("Missing server/app.py")
    else:
        # Check for main() function (flexible - with or without parameters)
        app_content = server_app.read_text(encoding="utf-8")
        if "def main(" not in app_content:
            issues.append("server/app.py missing main() function")

        # Check if main() is callable
        if not _has_main_guard_call(app_content):
            issues.append(
                "server/app.py main() function not callable (missing if __name__ == '__main__')"
            )

    return len(issues) == 0, issues


def get_deployment_modes(env_path: Path) -> dict[str, bool]:
    """
    Check which deployment modes are supported by the environment.

    Returns:
        `dict` mapping deployment mode names to whether they are supported.
    """
    modes = {
        "docker": False,
        "openenv_serve": False,
        "uv_run": False,
        "python_module": False,
    }

    # Check Docker (Dockerfile may be in server/ or at env root)
    modes["docker"] = (env_path / "server" / "Dockerfile").exists() or (
        env_path / "Dockerfile"
    ).exists()

    # Check multi-mode deployment readiness
    is_valid, _ = validate_multi_mode_deployment(env_path)
    if is_valid:
        modes["openenv_serve"] = True
        modes["uv_run"] = True
        modes["python_module"] = True

    return modes


def format_validation_report(env_name: str, is_valid: bool, issues: list[str]) -> str:
    """
    Format a validation report for display.

    Returns:
        `str`: formatted validation report.
    """
    if is_valid:
        return f"[OK] {env_name}: Ready for multi-mode deployment"

    report = [f"[FAIL] {env_name}: Not ready for multi-mode deployment", ""]
    report.append("Issues found:")
    for issue in issues:
        report.append(f"  - {issue}")

    return "\n".join(report)


def build_local_validation_json_report(
    env_name: str,
    env_path: Path,
    is_valid: bool,
    issues: list[str],
    deployment_modes: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Build a JSON report for local environment validation."""
    criteria = [
        _make_criterion(
            "multi_mode_deployment_readiness",
            "Environment structure is ready for multi-mode deployment",
            is_valid,
            details="No issues found" if is_valid else f"{len(issues)} issue(s) found",
            actual={"issues": issues},
        )
    ]

    if deployment_modes:
        for mode, supported in deployment_modes.items():
            criteria.append(
                _make_criterion(
                    f"deployment_mode_{mode}",
                    f"Deployment mode '{mode}' is supported",
                    supported,
                    required=False,
                )
            )

    return {
        "target": str(env_path),
        "environment": env_name,
        "validation_type": "local_environment",
        "standard_version": "local",
        "standard_profile": "openenv-local",
        "passed": is_valid,
        "summary": _build_summary(criteria),
        "criteria": criteria,
        "issues": issues,
        "deployment_modes": deployment_modes or {},
    }
