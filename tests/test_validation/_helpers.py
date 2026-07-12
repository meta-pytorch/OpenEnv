# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for validation architecture tests."""

from pathlib import Path


def write_valid_env(env_dir: Path) -> None:
    """Create a minimal source environment that passes static validation."""
    (env_dir / "server").mkdir(parents=True)
    (env_dir / "openenv.yaml").write_text(
        "spec_version: 1\n"
        "name: test_env\n"
        "type: space\n"
        "runtime: fastapi\n"
        "app: server.app:app\n"
        "port: 8000\n"
    )
    (env_dir / "uv.lock").write_text(
        'version = 1\nrevision = 1\nrequires-python = ">=3.10"\n'
    )
    (env_dir / "pyproject.toml").write_text(
        "[project]\n"
        'name = "test-env"\n'
        'version = "0.1.0"\n'
        'dependencies = ["openenv>=0.4.0"]\n'
        "\n"
        "[project.scripts]\n"
        'server = "server.app:main"\n'
    )
    (env_dir / "server" / "app.py").write_text(
        "def main():\n    return None\n\nif __name__ == '__main__':\n    main()\n"
    )
    (env_dir / "server" / "Dockerfile").write_text(
        'FROM python:3.12-slim\nCMD ["server"]\n'
    )


def write_harbor_task(task_root: Path) -> Path:
    """Create a multi-step Harbor task and return its environment directory."""
    environment = task_root / "environment"
    tests_dir = task_root / "tests"
    write_valid_env(environment)
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_root / "task.toml").write_text(
        'schema_version = "1.1"\n'
        "[[steps]]\n"
        'name = "grade"\n'
        "min_reward = { correctness = 0.8, style = 0.5 }\n"
        'artifacts = ["/workspace/result.json"]\n'
    )
    return environment
