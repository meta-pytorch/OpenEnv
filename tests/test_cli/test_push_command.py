# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end tests for push command."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from openenv_cli.commands.push import push_environment


@pytest.fixture
def mock_environment(tmp_path, monkeypatch):
    """Create a mock environment structure."""
    env_dir = tmp_path / "src" / "envs" / "test_env"
    server_dir = env_dir / "server"
    server_dir.mkdir(parents=True)
    
    # Create basic files
    (env_dir / "__init__.py").write_text("# Test")
    (env_dir / "README.md").write_text("# Test Environment")
    (server_dir / "Dockerfile").write_text("FROM test:latest")
    (server_dir / "app.py").write_text("# App")
    
    # Create core directory
    core_dir = tmp_path / "src" / "core"
    core_dir.mkdir(parents=True)
    (core_dir / "__init__.py").write_text("# Core")
    
    # Monkey patch Path to return our test paths
    monkeypatch.chdir(tmp_path)
    
    return tmp_path


class TestPushEnvironment:
    """Tests for push_environment function."""

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.space_exists")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.ensure_authenticated")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_full_workflow(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_ensure_auth,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_space_exists,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test full push workflow."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_ensure_auth.return_value = ("test_user", "test_token")
        mock_get_repo_id.return_value = "test_user/test_env"
        mock_space_exists.return_value = False
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push
        push_environment("test_env", namespace=None, private=False)
        
        # Verify calls
        mock_validate.assert_called_once_with("test_env")
        mock_ensure_auth.assert_called_once()
        mock_get_repo_id.assert_called_once_with("test_env", namespace=None, space_name=None)
        mock_space_exists.assert_called_once_with(mock_api, "test_user/test_env")
        mock_create_space.assert_called_once_with(mock_api, "test_user/test_env", private=False)
        mock_prepare_staging.assert_called_once()
        mock_copy_files.assert_called_once()
        mock_prepare_dockerfile.assert_called_once()
        mock_prepare_readme.assert_called_once()
        mock_upload.assert_called_once()

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.space_exists")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.ensure_authenticated")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_space_exists(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_ensure_auth,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_space_exists,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push when space already exists."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_ensure_auth.return_value = ("test_user", "test_token")
        mock_get_repo_id.return_value = "test_user/test_env"
        mock_space_exists.return_value = True  # Space already exists
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push
        push_environment("test_env")
        
        # Verify space was not created
        mock_create_space.assert_not_called()

    @patch("openenv_cli.commands.push.validate_environment")
    def test_push_environment_invalid_env(self, mock_validate, mock_environment):
        """Test push with invalid environment name."""
        mock_validate.side_effect = FileNotFoundError("Environment not found")
        
        with pytest.raises(FileNotFoundError, match="Environment not found"):
            push_environment("invalid_env")

    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.ensure_authenticated")
    def test_push_environment_auth_failure(self, mock_ensure_auth, mock_validate, mock_environment):
        """Test push when authentication fails."""
        mock_validate.return_value = Path("src/envs/test_env")
        mock_ensure_auth.side_effect = Exception("Authentication failed")
        
        with pytest.raises(Exception, match="Authentication failed"):
            push_environment("test_env")

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.space_exists")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.ensure_authenticated")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_with_namespace(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_ensure_auth,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_space_exists,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push with explicit namespace."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_ensure_auth.return_value = ("test_user", "test_token")
        mock_get_repo_id.return_value = "my-org/test_env"
        mock_space_exists.return_value = False
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with namespace
        push_environment("test_env", namespace="my-org")
        
        # Verify namespace was used
        mock_get_repo_id.assert_called_once_with("test_env", namespace="my-org", space_name=None)

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.space_exists")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.ensure_authenticated")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_private(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_ensure_auth,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_space_exists,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push with private space."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_ensure_auth.return_value = ("test_user", "test_token")
        mock_get_repo_id.return_value = "test_user/test_env"
        mock_space_exists.return_value = False
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with private flag
        push_environment("test_env", private=True)
        
        # Verify private space was created
        mock_create_space.assert_called_once_with(mock_api, "test_user/test_env", private=True)

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.space_exists")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.ensure_authenticated")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_with_space_name(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_ensure_auth,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_space_exists,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push with custom space name."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_ensure_auth.return_value = ("test_user", "test_token")
        mock_get_repo_id.return_value = "test_user/custom-space"
        mock_space_exists.return_value = False
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with space_name
        push_environment("test_env", space_name="custom-space")
        
        # Verify space_name was used
        mock_get_repo_id.assert_called_once_with("test_env", namespace=None, space_name="custom-space")
        mock_space_exists.assert_called_once_with(mock_api, "test_user/custom-space")
        mock_create_space.assert_called_once_with(mock_api, "test_user/custom-space", private=False)

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.space_exists")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.ensure_authenticated")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_with_namespace_and_space_name(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_ensure_auth,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_space_exists,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push with both namespace and space name."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_ensure_auth.return_value = ("test_user", "test_token")
        mock_get_repo_id.return_value = "my-org/custom-space"
        mock_space_exists.return_value = False
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with namespace and space_name
        push_environment("test_env", namespace="my-org", space_name="custom-space")
        
        # Verify both namespace and space_name were used
        mock_get_repo_id.assert_called_once_with("test_env", namespace="my-org", space_name="custom-space")
        mock_space_exists.assert_called_once_with(mock_api, "my-org/custom-space")
        mock_create_space.assert_called_once_with(mock_api, "my-org/custom-space", private=False)
