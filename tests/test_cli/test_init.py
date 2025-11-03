from pathlib import Path
from unittest.mock import patch, call

import pytest

from openenv_cli.commands.init import init as init_cmd


class TestInitCommand:
    @patch("subprocess.run")
    def test_init_creates_structure_and_git(self, mock_run, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        env_name = "my_env"
        init_cmd.callback = None  # ensure typer decorator side effects don't interfere
        init_cmd.__wrapped__ if hasattr(init_cmd, "__wrapped__") else None

        # Execute
        init_cmd(env_name=env_name, path=None, force=False)

        target = tmp_path / env_name
        assert target.exists()
        # Check key files
        assert (target / "openenv.yaml").exists()
        assert (target / "server" / "app.py").exists()
        assert (target / "server" / "Dockerfile").exists()
        assert (target / "server" / "requirements.txt").exists()
        assert (target / "models.py").exists()

        # Placeholder replacement
        manifest = (target / "openenv.yaml").read_text()
        assert "spec_version" in manifest
        assert env_name in manifest

        dockerfile = (target / "server" / "Dockerfile").read_text()
        assert "BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest" in dockerfile
        assert "ENABLE_WEB_INTERFACE" not in dockerfile

        # Git commands invoked
        expected_calls = [
            call(["git", "init"], cwd=str(target), check=True),
            call(["git", "add", "-A"], cwd=str(target), check=True),
            call(["git", "commit", "-m", "openenv: initial scaffold"], cwd=str(target), check=True),
        ]
        mock_run.assert_has_calls(expected_calls, any_order=False)

    def test_init_force_allows_non_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        env_name = "my_env"
        target = tmp_path / env_name
        target.mkdir(parents=True)
        (target / "EXISTING").write_text("x")

        with patch("subprocess.run") as mock_run:
            init_cmd(env_name=env_name, path=None, force=True)
            assert (target / "openenv.yaml").exists()


