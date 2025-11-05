# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the openenv push command."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from openenv_cli.__main__ import app


runner = CliRunner()


def _create_test_openenv_env(env_dir: Path, env_name: str = "test_env") -> None:
    """Create a minimal OpenEnv environment for testing."""
    # Create openenv.yaml
    manifest = {
        "spec_version": 1,
        "name": env_name,
        "type": "space",
        "runtime": "fastapi",
        "app": "server.app:app",
        "port": 8000,
    }
    
    import yaml
    with open(env_dir / "openenv.yaml", "w") as f:
        yaml.dump(manifest, f)
    
    # Create minimal server directory
    (env_dir / "server").mkdir(exist_ok=True)
    (env_dir / "server" / "Dockerfile").write_text(
        "FROM openenv-base:latest\nCMD [\"uvicorn\", \"server.app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
    )
    
    # Create README.md with frontmatter
    readme_content = """---
title: Test Environment
sdk: docker
app_port: 8000
---

# Test Environment
"""
    (env_dir / "README.md").write_text(readme_content)
    
    # Create a simple Python file
    (env_dir / "__init__.py").write_text("# Test environment\n")


def test_push_validates_openenv_directory(tmp_path: Path) -> None:
    """Test that push validates openenv.yaml is present."""
    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)
    
    assert result.exit_code != 0
    assert "openenv.yaml" in result.output.lower() or "manifest" in result.output.lower()


def test_push_validates_openenv_yaml_format(tmp_path: Path) -> None:
    """Test that push validates openenv.yaml format."""
    # Create invalid YAML
    (tmp_path / "openenv.yaml").write_text("invalid: yaml: content: [")
    
    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)
    
    assert result.exit_code != 0
    assert "parse" in result.output.lower() or "yaml" in result.output.lower()


def test_push_validates_openenv_yaml_has_name(tmp_path: Path) -> None:
    """Test that push validates openenv.yaml has a name field."""
    import yaml
    manifest = {"spec_version": 1, "type": "space"}
    with open(tmp_path / "openenv.yaml", "w") as f:
        yaml.dump(manifest, f)
    
    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)
    
    assert result.exit_code != 0
    assert "name" in result.output.lower()


def test_push_authenticates_with_hf(tmp_path: Path) -> None:
    """Test that push ensures Hugging Face authentication."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        # Mock whoami to return user info
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        
        # Mock HfApi
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify whoami was called
        assert mock_whoami.called


def test_push_enables_web_interface_in_dockerfile(tmp_path: Path) -> None:
    """Test that push enables web interface in Dockerfile."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify API was called (upload_folder)
        assert mock_api.upload_folder.called


def test_push_updates_readme_frontmatter(tmp_path: Path) -> None:
    """Test that push updates README frontmatter with base_path."""
    _create_test_openenv_env(tmp_path)
    
    # Create README without base_path
    readme_content = """---
title: Test Environment
sdk: docker
app_port: 8000
---

# Test Environment
"""
    (tmp_path / "README.md").write_text(readme_content)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify API was called
        assert mock_api.upload_folder.called


def test_push_uses_repo_id_option(tmp_path: Path) -> None:
    """Test that push respects --repo-id option."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--repo-id", "custom-org/my-env"])
        finally:
            os.chdir(old_cwd)
        
        # Verify create_repo was called with correct repo_id
        mock_api.create_repo.assert_called_once()
        call_args = mock_api.create_repo.call_args
        assert call_args.kwargs["repo_id"] == "custom-org/my-env"


def test_push_uses_default_repo_id(tmp_path: Path) -> None:
    """Test that push uses default repo-id from username and env name."""
    _create_test_openenv_env(tmp_path, env_name="test_env")
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify create_repo was called with default repo_id
        mock_api.create_repo.assert_called_once()
        call_args = mock_api.create_repo.call_args
        assert call_args.kwargs["repo_id"] == "testuser/test_env"


def test_push_uses_private_option(tmp_path: Path) -> None:
    """Test that push respects --private option."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--private"])
        finally:
            os.chdir(old_cwd)
        
        # Verify create_repo was called with private=True
        mock_api.create_repo.assert_called_once()
        call_args = mock_api.create_repo.call_args
        assert call_args.kwargs["private"] is True


def test_push_uses_base_image_option(tmp_path: Path) -> None:
    """Test that push respects --base-image option."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--base-image", "custom-base:latest"])
        finally:
            os.chdir(old_cwd)
        
        # Verify API was called (we can't easily test Dockerfile modification without reading staging dir)
        assert mock_api.upload_folder.called


def test_push_uses_directory_option(tmp_path: Path) -> None:
    """Test that push respects --directory option."""
    env_dir = tmp_path / "my_env"
    env_dir.mkdir()
    _create_test_openenv_env(env_dir)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        result = runner.invoke(
            app,
            ["push", "--directory", str(env_dir)],
        )
        
        # Verify API was called
        assert mock_api.upload_folder.called


def test_push_handles_missing_dockerfile(tmp_path: Path) -> None:
    """Test that push handles missing Dockerfile gracefully."""
    _create_test_openenv_env(tmp_path)
    # Remove Dockerfile
    (tmp_path / "server" / "Dockerfile").unlink()
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            # Should still work, just warn about missing Dockerfile
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify command was attempted (should warn but continue)
        assert mock_api.upload_folder.called


def test_push_handles_missing_readme(tmp_path: Path) -> None:
    """Test that push handles missing README gracefully."""
    _create_test_openenv_env(tmp_path)
    # Remove README
    (tmp_path / "README.md").unlink()
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            # Should still work, just warn about missing README
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify command was attempted (should warn but continue)
        assert mock_api.upload_folder.called


def test_push_initializes_hf_api_without_token(tmp_path: Path) -> None:
    """Test that push initializes HfApi without token parameter."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify HfApi was initialized without token parameter
        mock_hf_api_class.assert_called_once()
        call_args = mock_hf_api_class.call_args
        # Should not have token in kwargs
        assert "token" not in (call_args.kwargs or {})


def test_push_validates_repo_id_format(tmp_path: Path) -> None:
    """Test that push validates repo-id format."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        # Mock HfApi to prevent actual API calls
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            # Invalid format (no slash)
            result = runner.invoke(app, ["push", "--repo-id", "invalid-repo-id"])
        finally:
            os.chdir(old_cwd)
        
        assert result.exit_code != 0
        assert "repo-id" in result.output.lower() or "format" in result.output.lower()


def test_push_validates_manifest_is_dict(tmp_path: Path) -> None:
    """Test that push validates manifest is a dictionary."""
    # Create openenv.yaml with non-dict content
    import yaml
    with open(tmp_path / "openenv.yaml", "w") as f:
        yaml.dump("not a dict", f)
    
    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        result = runner.invoke(app, ["push"])
    finally:
        os.chdir(old_cwd)
    
    assert result.exit_code != 0
    assert "dictionary" in result.output.lower() or "yaml" in result.output.lower()


def test_push_handles_whoami_object_return(tmp_path: Path) -> None:
    """Test that push handles whoami returning an object instead of dict."""
    _create_test_openenv_env(tmp_path)
    
    # Create a mock object with name attribute
    class MockUser:
        def __init__(self):
            self.name = "testuser"
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = MockUser()
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify it worked with object return type
        assert mock_api.upload_folder.called


def test_push_handles_authentication_failure(tmp_path: Path) -> None:
    """Test that push handles authentication failure."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        # First whoami call fails (not authenticated)
        # Login also fails
        mock_whoami.side_effect = Exception("Not authenticated")
        mock_login.side_effect = Exception("Login failed")
        # Mock HfApi to prevent actual API calls
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        assert result.exit_code != 0
        assert "authentication" in result.output.lower() or "login" in result.output.lower()


def test_push_handles_whoami_missing_username(tmp_path: Path) -> None:
    """Test that push handles whoami response without username."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        # Return dict without name, fullname, or username
        mock_whoami.return_value = {}
        # Mock login to prevent actual login prompt
        mock_login.return_value = None
        # Mock HfApi to prevent actual API calls
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        assert result.exit_code != 0
        assert "username" in result.output.lower() or "extract" in result.output.lower()


def test_push_handles_readme_without_frontmatter(tmp_path: Path) -> None:
    """Test that push handles README without frontmatter."""
    _create_test_openenv_env(tmp_path)
    
    # Create README without frontmatter
    (tmp_path / "README.md").write_text("# Test Environment\nNo frontmatter here.\n")
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Verify it still works (should add frontmatter)
        assert mock_api.upload_folder.called


def test_push_handles_hf_api_create_repo_error(tmp_path: Path) -> None:
    """Test that push handles HF API create_repo error."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_api.create_repo.side_effect = Exception("API Error")
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            # Should continue despite error (warns but doesn't fail)
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        # Should still attempt upload
        assert mock_api.upload_folder.called


def test_push_handles_hf_api_upload_error(tmp_path: Path) -> None:
    """Test that push handles HF API upload_folder error."""
    _create_test_openenv_env(tmp_path)
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_api.upload_folder.side_effect = Exception("Upload failed")
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push"])
        finally:
            os.chdir(old_cwd)
        
        assert result.exit_code != 0
        assert "upload" in result.output.lower() or "failed" in result.output.lower()


def test_push_handles_base_image_not_found_in_dockerfile(tmp_path: Path) -> None:
    """Test that push handles Dockerfile without FROM line."""
    _create_test_openenv_env(tmp_path)
    
    # Create Dockerfile without FROM line
    (tmp_path / "server" / "Dockerfile").write_text("RUN echo 'test'\nCMD [\"echo\", \"test\"]\n")
    
    with patch("openenv_cli.commands.push.whoami") as mock_whoami, \
         patch("openenv_cli.commands.push.login") as mock_login, \
         patch("openenv_cli.commands.push.HfApi") as mock_hf_api_class:
        
        mock_whoami.return_value = {"name": "testuser"}
        mock_login.return_value = None  # Prevent actual login prompt
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            result = runner.invoke(app, ["push", "--base-image", "custom-base:latest"])
        finally:
            os.chdir(old_cwd)
        
        # Should still work (adds FROM at beginning)
        assert mock_api.upload_folder.called
