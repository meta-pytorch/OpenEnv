# SPDX-License-Identifier: BSD-3-Clause

"""Tests for openenv build helpers."""

from pathlib import Path

from openenv.cli.commands.build import _detect_build_context


def test_detect_build_context_uses_envs_child_as_in_repo(tmp_path: Path) -> None:
    """An environment below repo_root/envs uses the repository as build context."""
    repo_root = tmp_path / "repo"
    env_path = repo_root / "envs" / "example_env"
    env_path.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    assert _detect_build_context(env_path) == (
        "in-repo",
        repo_root.absolute(),
        repo_root.absolute(),
    )


def test_detect_build_context_keeps_repo_sibling_standalone(tmp_path: Path) -> None:
    """A repo-local directory outside envs/ remains a standalone environment."""
    repo_root = tmp_path / "repo"
    env_path = repo_root / "examples" / "example_env"
    env_path.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    assert _detect_build_context(env_path) == (
        "standalone",
        env_path.absolute(),
        None,
    )


def test_detect_build_context_keeps_non_git_path_standalone(tmp_path: Path) -> None:
    """A path outside any repository remains a standalone environment."""
    env_path = tmp_path / "example_env"
    env_path.mkdir()

    assert _detect_build_context(env_path) == (
        "standalone",
        env_path.absolute(),
        None,
    )
