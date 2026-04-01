"""Regression tests for BrowserGym packaging constraints."""

from __future__ import annotations

from pathlib import Path
import tomllib


def test_browsergym_default_dependencies_match_miniwob_runtime() -> None:
    """Default installs should avoid the known WebArena/playwright resolver conflict."""
    pyproject_path = (
        Path(__file__).resolve().parents[2] / "envs" / "browsergym_env" / "pyproject.toml"
    )
    data = tomllib.loads(pyproject_path.read_text())
    dependencies = data["project"]["dependencies"]

    assert "browsergym-core==0.14.3" in dependencies
    assert "browsergym-miniwob==0.14.3" in dependencies
    assert "playwright==1.44.0" in dependencies
    assert not any(dep.startswith("browsergym-webarena") for dep in dependencies)
