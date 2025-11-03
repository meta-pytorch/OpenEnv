from pathlib import Path
import pytest

from openenv_cli.utils.env_loader import (
    validate_environment_at,
    resolve_environment,
)


def test_validate_environment_at_nonexistent(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        validate_environment_at(missing)


def test_validate_environment_at_not_directory(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(FileNotFoundError):
        validate_environment_at(f)


def test_validate_environment_at_missing_server(tmp_path):
    env_root = tmp_path / "env"
    env_root.mkdir()
    with pytest.raises(FileNotFoundError, match="server/"):
        validate_environment_at(env_root)


def test_resolve_environment_with_env_path(tmp_path, monkeypatch):
    env_root = tmp_path / "myenv"
    server = env_root / "server"
    server.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    name, root = resolve_environment(env_name=None, env_path=str(env_root))
    assert name == "myenv"
    assert root == env_root


def test_resolve_environment_with_env_name_repo_structure(tmp_path, monkeypatch):
    # create src/envs/name structure
    env_root = tmp_path / "src" / "envs" / "abc"
    (env_root / "server").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    name, root = resolve_environment(env_name="abc", env_path=None)
    assert name == "abc"
    assert root == env_root


