# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Validation utilities for multi-mode deployment readiness.

This module provides functions to check if environments are properly
configured for multi-mode deployment (Docker, direct Python, notebooks, clusters).
"""

import subprocess
import tomllib
from pathlib import Path


def validate_multi_mode_deployment(env_path: Path) -> tuple[bool, list[str]]:
    """
    Validate that an environment is ready for multi-mode deployment.

    Checks:
    1. pyproject.toml exists
    2. uv.lock exists and is up-to-date
    3. pyproject.toml has [project.scripts] with server entry point
    4. server/app.py has a main() function
    5. Required dependencies are present

    Returns:
        Tuple of (is_valid, list of issues found)
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
    else:
        # Check if uv.lock is up-to-date (optional, can be expensive)
        # We can add a check using `uv lock --check` if needed
        try:
            result = subprocess.run(
                ["uv", "lock", "--check", "--directory", str(env_path)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                issues.append("uv.lock is out of date with pyproject.toml - run 'uv lock' to update")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # If uv is not available or times out, skip this check
            pass

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
    has_openenv = any(dep.startswith("openenv") and not dep.startswith("openenv-core") for dep in deps)
    has_legacy_core = any(dep.startswith("openenv-core") for dep in deps)

    if not (has_openenv or has_legacy_core):
        issues.append("Missing required dependency: openenv>=0.2.0")
    elif has_legacy_core and not has_openenv:
        issues.append("Dependency on openenv-core is deprecated; use openenv>=0.2.0 instead")

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
        if "__name__" not in app_content or "main()" not in app_content:
            issues.append(
                "server/app.py main() function not callable (missing if __name__ == '__main__')"
            )

    return len(issues) == 0, issues


def get_deployment_modes(env_path: Path) -> dict[str, bool]:
    """
    Check which deployment modes are supported by the environment.

    Returns:
        Dictionary with deployment mode names and whether they're supported
    """
    modes = {
        "docker": False,
        "openenv_serve": False,
        "uv_run": False,
        "python_module": False,
    }

    # Check Docker
    dockerfile = env_path / "server" / "Dockerfile"
    modes["docker"] = dockerfile.exists()

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
        Formatted report string
    """
    if is_valid:
        return f"[OK] {env_name}: Ready for multi-mode deployment"

    report = [f"[FAIL] {env_name}: Not ready for multi-mode deployment", ""]
    report.append("Issues found:")
    for issue in issues:
        report.append(f"  - {issue}")

    return "\n".join(report)
