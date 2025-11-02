# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for builder module."""

import os
from unittest.mock import patch

import pytest

from openenv_cli.core.builder import (
    prepare_staging_directory,
    prepare_dockerfile,
    prepare_readme,
)


@pytest.fixture
def test_env_path(tmp_path):
    """Create a temporary test environment."""
    env_dir = tmp_path / "src" / "envs" / "test_env"
    server_dir = env_dir / "server"
    server_dir.mkdir(parents=True)
    
    # Create basic environment files
    (env_dir / "__init__.py").write_text("# Test env")
    (env_dir / "models.py").write_text("# Test models")
    (env_dir / "client.py").write_text("# Test client")
    (env_dir / "README.md").write_text("# Test Environment\n\nTest description")
    
    # Create Dockerfile
    (server_dir / "Dockerfile").write_text(
        "ARG BASE_IMAGE=openenv-base:latest\n"
        "FROM ${BASE_IMAGE}\n"
        "COPY src/core/ /app/src/core/\n"
        "COPY src/envs/test_env/ /app/src/envs/test_env/\n"
        "CMD [\"uvicorn\", \"envs.test_env.server.app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]"
    )
    
    # Create app.py
    (server_dir / "app.py").write_text("# Test app")
    
    return tmp_path


@pytest.fixture
def staging_dir(tmp_path):
    """Create staging directory for tests."""
    staging = tmp_path / "staging"
    staging.mkdir()
    return staging


@pytest.fixture
def repo_root(tmp_path, test_env_path):
    """Create a mock repo root structure."""
    # Create core directory (test_env_path already created src/envs)
    core_dir = tmp_path / "src" / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    if not (core_dir / "__init__.py").exists():
        (core_dir / "__init__.py").write_text("# Core")
    
    # Create envs directory structure (may already exist from test_env_path)
    envs_dir = tmp_path / "src" / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    
    return tmp_path


class TestPrepareStagingDirectory:
    """Tests for prepare_staging_directory function."""

    @patch("openenv_cli.core.builder.Path")
    def test_prepare_staging_directory_creates_structure(self, mock_path, staging_dir):
        """Test that staging directory structure is created."""
        # This will be tested through integration with copy functions
        pass


class TestCopyEnvironmentFiles:
    """Tests for copy_environment_files function."""

    def test_copy_environment_files_copies_files(self, repo_root, tmp_path, monkeypatch):
        """Test that environment files are copied correctly."""
        from openenv_cli.core.builder import copy_environment_files, prepare_staging_directory
        
        # Set up test environment (test_env_path already created the directory)
        test_env_dir = repo_root / "src" / "envs" / "test_env"
        test_env_dir.mkdir(parents=True, exist_ok=True)
        (test_env_dir / "models.py").write_text("# Test")
        
        # Create core directory (repo_root already created it)
        core_dir = repo_root / "src" / "core"
        core_dir.mkdir(parents=True, exist_ok=True)
        if not (core_dir / "__init__.py").exists():
            (core_dir / "__init__.py").write_text("# Core")
        
        # Prepare staging directory (creates the structure)
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        # Set working directory
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            copy_environment_files("test_env", staging_dir)
            
            # Verify files were copied
            assert (staging_dir / "src" / "core" / "__init__.py").exists()
            assert (staging_dir / "src" / "envs" / "test_env" / "models.py").exists()
        finally:
            os.chdir(old_cwd)


class TestPrepareDockerfile:
    """Tests for prepare_dockerfile function."""

    def test_prepare_dockerfile_uses_existing_dockerfile(self, repo_root, tmp_path, monkeypatch):
        """Test that existing Dockerfile is used and modified."""
        
        # Create test environment with Dockerfile (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / "test_env" / "server"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "Dockerfile").write_text(
            "ARG BASE_IMAGE=openenv-base:latest\n"
            "FROM ${BASE_IMAGE}\n"
            "COPY src/core/ /app/src/core/\n"
            "CMD [\"uvicorn\", \"envs.test_env.server.app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]"
        )
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            base_image = "ghcr.io/meta-pytorch/openenv-base:latest"
            prepare_dockerfile("test_env", staging_dir, base_image)
            
            # Check Dockerfile was created
            dockerfile_path = staging_dir / "Dockerfile"
            assert dockerfile_path.exists()
            
            # Check base image was replaced
            content = dockerfile_path.read_text()
            assert base_image in content
            assert "ENV ENABLE_WEB_INTERFACE=true" in content or "ENABLE_WEB_INTERFACE=true" in content
        finally:
            os.chdir(old_cwd)

    def test_prepare_dockerfile_creates_default_if_missing(self, repo_root, tmp_path, monkeypatch):
        """Test that default Dockerfile is created if env doesn't have one."""
        
        # Create test environment without Dockerfile (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "models.py").write_text("# Test")
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            base_image = "ghcr.io/meta-pytorch/openenv-base:latest"
            prepare_dockerfile("test_env", staging_dir, base_image)
            
            # Check Dockerfile was created
            dockerfile_path = staging_dir / "Dockerfile"
            assert dockerfile_path.exists()
            
            content = dockerfile_path.read_text()
            assert base_image in content
            assert "test_env" in content
        finally:
            os.chdir(old_cwd)


class TestPrepareReadme:
    """Tests for prepare_readme function."""

    def test_prepare_readme_adds_front_matter(self, repo_root, tmp_path, monkeypatch):
        """Test that README gets HF front matter added."""
        
        # Create test environment with README (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "README.md").write_text(
            "# Test Environment\n\nThis is a test environment."
        )
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            prepare_readme("test_env", staging_dir)
            
            # Check README was created
            readme_path = staging_dir / "README.md"
            assert readme_path.exists()
            
            content = readme_path.read_text()
            # Check for HF front matter
            assert "---" in content
            assert "sdk: docker" in content
            assert "Test Environment" in content
        finally:
            os.chdir(old_cwd)

    def test_prepare_readme_handles_existing_front_matter(self, repo_root, tmp_path, monkeypatch):
        """Test that README with existing front matter is used as-is."""
        
        # Create test environment with README that has front matter (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True, exist_ok=True)
        original_content = (
            "---\n"
            "title: Test Environment\n"
            "emoji: 🎮\n"
            "colorFrom: red\n"
            "colorTo: blue\n"
            "sdk: docker\n"
            "---\n"
            "# Test Environment\n\nThis is a test environment."
        )
        (env_dir / "README.md").write_text(original_content)
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            prepare_readme("test_env", staging_dir)
            
            # Check README was created
            readme_path = staging_dir / "README.md"
            assert readme_path.exists()
            
            content = readme_path.read_text()
            # Should use original content as-is
            assert content == original_content
            assert "emoji: 🎮" in content
            assert "colorFrom: red" in content
            assert "colorTo: blue" in content
        finally:
            os.chdir(old_cwd)

    def test_prepare_readme_generates_front_matter_when_missing(self, repo_root, tmp_path, monkeypatch):
        """Test that README without front matter gets generated front matter."""
        
        # Create test environment with README without front matter (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "README.md").write_text(
            "# Test Environment\n\nThis is a test environment."
        )
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            prepare_readme("test_env", staging_dir)
            
            # Check README was created
            readme_path = staging_dir / "README.md"
            assert readme_path.exists()
            
            content = readme_path.read_text()
            # Should have generated front matter
            assert content.startswith("---")
            assert "sdk: docker" in content
            assert "emoji:" in content
            assert "colorFrom:" in content
            assert "colorTo:" in content
            # Should include original content
            assert "# Test Environment" in content
            assert "This is a test environment." in content
        finally:
            os.chdir(old_cwd)

    def test_prepare_readme_uses_server_readme_front_matter(self, repo_root, tmp_path, monkeypatch):
        """Test that server/README.md front matter is used when present."""
        
        # Create test environment (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True, exist_ok=True)
        server_dir = env_dir / "server"
        server_dir.mkdir(parents=True, exist_ok=True)
        
        # Main README without front matter
        (env_dir / "README.md").write_text("# Test Environment\n\nDescription.")
        
        # Server README with front matter
        server_readme_content = (
            "---\n"
            "title: Server Test\n"
            "emoji: 🚀\n"
            "colorFrom: purple\n"
            "colorTo: indigo\n"
            "sdk: docker\n"
            "---\n"
            "# Server README\n\nServer-specific content."
        )
        (server_dir / "README.md").write_text(server_readme_content)
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            prepare_readme("test_env", staging_dir)
            
            # Check README was created
            readme_path = staging_dir / "README.md"
            assert readme_path.exists()
            
            content = readme_path.read_text()
            # Should use server README content as-is
            assert content == server_readme_content
            assert "emoji: 🚀" in content
            assert "colorFrom: purple" in content
            assert "colorTo: indigo" in content
        finally:
            os.chdir(old_cwd)

    def test_prepare_readme_handles_empty_env_name(self, repo_root, tmp_path, monkeypatch):
        """Test that README generation handles empty env_name safely."""
        
        # Create test environment (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / ""
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / "README.md").write_text(
            "# Test Environment\n\nThis is a test environment."
        )
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            # Should not raise IndexError
            prepare_readme("", staging_dir)
            
            # Check README was created
            readme_path = staging_dir / "README.md"
            assert readme_path.exists()
            
            content = readme_path.read_text()
            # Should have generated front matter
            assert content.startswith("---")
            assert "sdk: docker" in content
            # Title should handle empty string
            assert "title:" in content
        finally:
            os.chdir(old_cwd)

    def test_prepare_readme_no_duplicate_when_original_has_front_matter(self, repo_root, tmp_path, monkeypatch):
        """Test that original README with front matter is not appended (no duplicates)."""
        
        # Create test environment with README that has front matter (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True, exist_ok=True)
        original_content = (
            "---\n"
            "title: Test Environment\n"
            "emoji: 🎮\n"
            "colorFrom: red\n"
            "colorTo: blue\n"
            "sdk: docker\n"
            "---\n"
            "# Test Environment\n\nThis is a test environment.\n\n## About\n\nSome custom about content."
        )
        (env_dir / "README.md").write_text(original_content)
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            prepare_readme("test_env", staging_dir)
            
            # Check README was created
            readme_path = staging_dir / "README.md"
            assert readme_path.exists()
            
            content = readme_path.read_text()
            # Should use original content as-is (not append duplicate sections)
            assert content == original_content
            # Should not have duplicate "About" sections
            assert content.count("## About") == 1
        finally:
            os.chdir(old_cwd)

    def test_prepare_readme_appends_original_without_front_matter(self, repo_root, tmp_path, monkeypatch):
        """Test that original README without front matter is appended correctly."""
        
        # Create test environment with README without front matter (test_env_path already created the directory)
        env_dir = repo_root / "src" / "envs" / "test_env"
        env_dir.mkdir(parents=True, exist_ok=True)
        original_content = (
            "# Test Environment\n\n"
            "This is a test environment with custom documentation.\n\n"
            "## Custom Section\n\n"
            "This is environment-specific content that should be preserved."
        )
        (env_dir / "README.md").write_text(original_content)
        
        # Prepare staging directory
        staging_dir = prepare_staging_directory("test_env", "test:latest", str(tmp_path / "hf-staging"))
        
        old_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            prepare_readme("test_env", staging_dir)
            
            # Check README was created
            readme_path = staging_dir / "README.md"
            assert readme_path.exists()
            
            content = readme_path.read_text()
            # Should have generated front matter
            assert content.startswith("---")
            # Should have generated standard sections
            assert "## About" in content
            assert "## Web Interface" in content
            # Should append original content after generated sections
            assert "Custom Section" in content
            assert "environment-specific content" in content
            # Should not duplicate the title - original "# Test Environment" appears once (from appended content)
            assert content.count("# Test Environment") == 1
        finally:
            os.chdir(old_cwd)
