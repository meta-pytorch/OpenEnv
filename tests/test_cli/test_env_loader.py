# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for environment loader utilities."""

from pathlib import Path

import pytest

from openenv_cli.utils.env_loader import load_env_metadata, validate_environment


class TestValidateEnvironment:
    """Tests for validate_environment function."""

    def test_validate_environment_success(self, tmp_path, monkeypatch):
        """Test validate_environment with valid environment."""
        env_dir = tmp_path / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True)
        
        monkeypatch.chdir(tmp_path)
        
        result = validate_environment("test_env")
        
        # Result is relative path, but should exist and be a directory
        assert result == Path("src/envs/test_env")
        assert result.exists()
        assert result.is_dir()

    def test_validate_environment_not_found(self, tmp_path, monkeypatch):
        """Test validate_environment when environment doesn't exist."""
        monkeypatch.chdir(tmp_path)
        
        with pytest.raises(FileNotFoundError, match="Environment 'missing_env' not found"):
            validate_environment("missing_env")

    def test_validate_environment_not_directory(self, tmp_path, monkeypatch):
        """Test validate_environment when path exists but is not a directory."""
        env_file = tmp_path / "src" / "envs" / "test_env"
        env_file.parent.mkdir(parents=True)
        env_file.write_text("not a directory")
        
        monkeypatch.chdir(tmp_path)
        
        with pytest.raises(FileNotFoundError, match="Environment 'test_env' is not a directory"):
            validate_environment("test_env")


class TestLoadEnvMetadata:
    """Tests for load_env_metadata function."""

    def test_load_env_metadata_basic(self, tmp_path, monkeypatch):
        """Test load_env_metadata with basic environment."""
        env_dir = tmp_path / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True)
        
        monkeypatch.chdir(tmp_path)
        
        metadata = load_env_metadata("test_env")
        
        assert metadata["name"] == "test_env"
        # Path is returned as relative string
        assert metadata["path"] == "src/envs/test_env"
        assert "readme" not in metadata
        assert "has_server" not in metadata

    def test_load_env_metadata_with_readme(self, tmp_path, monkeypatch):
        """Test load_env_metadata with README."""
        env_dir = tmp_path / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True)
        readme = env_dir / "README.md"
        readme.write_text("# Test Environment\n\nThis is a test.")
        
        monkeypatch.chdir(tmp_path)
        
        metadata = load_env_metadata("test_env")
        
        assert metadata["name"] == "test_env"
        assert "readme" in metadata
        assert metadata["readme"] == "# Test Environment\n\nThis is a test."
        assert metadata["title"] == "Test Environment"

    def test_load_env_metadata_with_server(self, tmp_path, monkeypatch):
        """Test load_env_metadata with server directory."""
        env_dir = tmp_path / "src" / "envs" / "test_env"
        server_dir = env_dir / "server"
        server_dir.mkdir(parents=True)
        
        monkeypatch.chdir(tmp_path)
        
        metadata = load_env_metadata("test_env")
        
        assert metadata["has_server"] is True
        assert metadata.get("has_dockerfile") is None

    def test_load_env_metadata_with_dockerfile(self, tmp_path, monkeypatch):
        """Test load_env_metadata with Dockerfile."""
        env_dir = tmp_path / "src" / "envs" / "test_env"
        server_dir = env_dir / "server"
        server_dir.mkdir(parents=True)
        dockerfile = server_dir / "Dockerfile"
        dockerfile.write_text("FROM test:latest")
        
        monkeypatch.chdir(tmp_path)
        
        metadata = load_env_metadata("test_env")
        
        assert metadata["has_server"] is True
        assert metadata["has_dockerfile"] is True
        # Path is returned as relative string
        assert metadata["dockerfile_path"] == "src/envs/test_env/server/Dockerfile"

    def test_load_env_metadata_with_models(self, tmp_path, monkeypatch):
        """Test load_env_metadata with models.py."""
        env_dir = tmp_path / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True)
        models = env_dir / "models.py"
        models.write_text("# Models")
        
        monkeypatch.chdir(tmp_path)
        
        metadata = load_env_metadata("test_env")
        
        assert metadata["has_models"] is True

    def test_load_env_metadata_with_client(self, tmp_path, monkeypatch):
        """Test load_env_metadata with client.py."""
        env_dir = tmp_path / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True)
        client = env_dir / "client.py"
        client.write_text("# Client")
        
        monkeypatch.chdir(tmp_path)
        
        metadata = load_env_metadata("test_env")
        
        assert metadata["has_client"] is True

    def test_load_env_metadata_full(self, tmp_path, monkeypatch):
        """Test load_env_metadata with all components."""
        env_dir = tmp_path / "src" / "envs" / "test_env"
        server_dir = env_dir / "server"
        server_dir.mkdir(parents=True)
        
        # Create all files
        (env_dir / "README.md").write_text("# Full Environment\n\nComplete setup.")
        (env_dir / "models.py").write_text("# Models")
        (env_dir / "client.py").write_text("# Client")
        (server_dir / "Dockerfile").write_text("FROM test:latest")
        
        monkeypatch.chdir(tmp_path)
        
        metadata = load_env_metadata("test_env")
        
        assert metadata["name"] == "test_env"
        assert "readme" in metadata
        assert metadata["title"] == "Full Environment"
        assert metadata["has_server"] is True
        assert metadata["has_dockerfile"] is True
        assert metadata["has_models"] is True
        assert metadata["has_client"] is True

