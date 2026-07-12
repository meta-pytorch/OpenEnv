# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for validation architecture tests."""

from pathlib import Path


def write_harbor_task(task_root: Path) -> Path:
    """Create a multi-step Harbor task and return its environment directory."""
    environment = task_root / "environment"
    tests_dir = task_root / "tests"
    environment.mkdir(parents=True)
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
