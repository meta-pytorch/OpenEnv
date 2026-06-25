# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for openenv build helpers."""

from pathlib import Path

from openenv.cli.commands.build import _detect_build_context, _parse_build_args


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


def test_parse_build_args_preserves_values_with_equals() -> None:
    """Build arg parsing only splits on the first equals sign."""
    assert _parse_build_args(["ENV=prod", "TOKEN=a=b=c"]) == {
        "ENV": "prod",
        "TOKEN": "a=b=c",
    }


def test_parse_build_args_warns_and_skips_invalid(capsys) -> None:
    """Malformed build args keep the existing warning-and-skip behavior."""
    assert _parse_build_args(["missing_equals", "ENV=prod"]) == {"ENV": "prod"}

    captured = capsys.readouterr()
    assert "Warning: Invalid build arg format: missing_equals" in captured.err
