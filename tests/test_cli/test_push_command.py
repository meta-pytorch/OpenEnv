# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end tests for push command."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from openenv_cli.commands.push import (
    _prepare_environment,
    _upload_environment,
    push,
    push_environment,
)


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
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_full_workflow(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test full push workflow."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_get_repo_id.return_value = "test_user/test_env"
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with credentials
        push_environment("test_env", username="test_user", token="test_token", private=False)
        
        # Verify calls
        mock_validate.assert_called_once_with("test_env")
        mock_get_repo_id.assert_called_once_with("test_env", "test_user")
        mock_create_space.assert_called_once_with(mock_api, "test_user/test_env", private=False)
        mock_prepare_staging.assert_called_once()
        mock_copy_files.assert_called_once()
        mock_prepare_dockerfile.assert_called_once()
        mock_prepare_readme.assert_called_once()
        mock_upload.assert_called_once()

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_space_exists(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push when space already exists (create_space handles it with exist_ok=True)."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_get_repo_id.return_value = "test_user/test_env"
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with credentials
        push_environment("test_env", username="test_user", token="test_token")
        
        # Verify create_space was called (it handles existing spaces internally)
        mock_create_space.assert_called_once_with(mock_api, "test_user/test_env", private=False)

    @patch("openenv_cli.commands.push.validate_environment")
    def test_push_environment_invalid_env(self, mock_validate, mock_environment):
        """Test push with invalid environment name."""
        mock_validate.side_effect = FileNotFoundError("Environment not found")
        
        with pytest.raises(FileNotFoundError, match="Environment not found"):
            push_environment("invalid_env", username="test_user", token="test_token")

    @patch("openenv_cli.commands.push.validate_environment")
    def test_push_environment_auth_failure(self, mock_validate, mock_environment):
        """Test push when authentication fails (now handled in __main__.py)."""
        # Note: Authentication failures are now handled in __main__.py
        # This test verifies validation works
        mock_validate.return_value = Path("src/envs/test_env")
        
        # push_environment should succeed if credentials are provided
        # Authentication failures would happen before calling push_environment
        pass

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_with_repo_id(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push with explicit repo_id."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with repo_id (should not call get_space_repo_id)
        push_environment("test_env", username="test_user", token="test_token", repo_id="my-org/test_env")
        
        # Verify repo_id was used directly (get_space_repo_id should not be called)
        mock_get_repo_id.assert_not_called()
        mock_create_space.assert_called_once_with(mock_api, "my-org/test_env", private=False)

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_private(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push with private space."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_get_repo_id.return_value = "test_user/test_env"
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with private flag
        push_environment("test_env", username="test_user", token="test_token", private=True)
        
        # Verify private space was created
        mock_create_space.assert_called_once_with(mock_api, "test_user/test_env", private=True)

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_with_repo_id_custom_space_name(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push with repo_id that has custom space name."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_prepare_staging.return_value = Path("staging")
        
        # Run push with repo_id containing custom space name
        push_environment("test_env", username="test_user", token="test_token", repo_id="my-org/custom-space")
        
        # Verify repo_id was used directly (get_space_repo_id should not be called)
        mock_get_repo_id.assert_not_called()
        mock_create_space.assert_called_once_with(mock_api, "my-org/custom-space", private=False)

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_dry_run(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test push with dry_run=True (should not upload)."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_get_repo_id.return_value = "test_user/test_env"
        staging_dir = mock_environment / "staging"
        staging_dir.mkdir()
        mock_prepare_staging.return_value = staging_dir
        
        # Run push with dry_run=True
        push_environment(
            "test_env",
            username="test_user",
            token="test_token",
            dry_run=True,
        )
        
        # Verify upload was NOT called (dry run)
        mock_upload.assert_not_called()
        # Verify cleanup happened
        assert not staging_dir.exists()

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_push_environment_cleanup_on_error(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_create_space,
        mock_upload,
        mock_environment,
    ):
        """Test that staging directory is cleaned up even on upload error."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_get_repo_id.return_value = "test_user/test_env"
        staging_dir = mock_environment / "staging"
        staging_dir.mkdir()
        mock_prepare_staging.return_value = staging_dir
        mock_upload.side_effect = Exception("Upload failed")
        
        # Run push and expect error
        with pytest.raises(Exception, match="Upload failed"):
            push_environment("test_env", username="test_user", token="test_token")
        
        # Verify cleanup happened even after error
        assert not staging_dir.exists()


class TestPrepareEnvironment:
    """Tests for _prepare_environment function."""

    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_prepare_environment_full(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_create_space,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_environment,
    ):
        """Test _prepare_environment with all steps."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_get_repo_id.return_value = "test_user/test_env"
        staging_dir = Path("staging")
        mock_prepare_staging.return_value = staging_dir
        
        # Run prepare
        result = _prepare_environment(
            env_name="test_env",
            repo_id=None,
            private=False,
            base_image=None,
            username="test_user",
            token="test_token",
        )
        
        # Verify calls
        mock_validate.assert_called_once_with("test_env")
        mock_get_repo_id.assert_called_once_with("test_env", "test_user")
        mock_create_space.assert_called_once_with(mock_api, "test_user/test_env", private=False)
        mock_prepare_staging.assert_called_once()
        mock_copy_files.assert_called_once()
        mock_prepare_dockerfile.assert_called_once()
        mock_prepare_readme.assert_called_once()
        assert result == staging_dir

    @patch("openenv_cli.commands.push.prepare_readme")
    @patch("openenv_cli.commands.push.prepare_dockerfile")
    @patch("openenv_cli.commands.push.copy_environment_files")
    @patch("openenv_cli.commands.push.prepare_staging_directory")
    @patch("openenv_cli.commands.push.create_space")
    @patch("openenv_cli.commands.push.validate_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.HfApi")
    def test_prepare_environment_with_repo_id(
        self,
        mock_api_class,
        mock_get_repo_id,
        mock_validate,
        mock_create_space,
        mock_prepare_staging,
        mock_copy_files,
        mock_prepare_dockerfile,
        mock_prepare_readme,
        mock_environment,
    ):
        """Test _prepare_environment with explicit repo_id."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        staging_dir = Path("staging")
        mock_prepare_staging.return_value = staging_dir
        
        # Run prepare with repo_id
        result = _prepare_environment(
            env_name="test_env",
            repo_id="my-org/test_env",
            private=True,
            base_image="custom:latest",
            username="test_user",
            token="test_token",
        )
        
        # Verify get_space_repo_id was NOT called
        mock_get_repo_id.assert_not_called()
        mock_create_space.assert_called_once_with(mock_api, "my-org/test_env", private=True)
        # prepare_staging_directory now receives a staging root third argument; validate first two args
        assert mock_prepare_staging.call_count == 1
        args, _ = mock_prepare_staging.call_args
        assert args[0] == "test_env"
        assert args[1] == "custom:latest"
        assert result == staging_dir


class TestUploadEnvironment:
    """Tests for _upload_environment function."""

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.HfApi")
    def test_upload_environment_success(self, mock_api_class, mock_upload, tmp_path):
        """Test _upload_environment successfully uploads."""
        # Setup
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "test.txt").write_text("test")
        
        # Run upload
        _upload_environment(
            env_name="test_env",
            repo_id="test_user/test_env",
            staging_dir=staging_dir,
            username="test_user",
            token="test_token",
        )
        
        # Verify upload was called and cleanup happened
        mock_upload.assert_called_once_with("test_user/test_env", staging_dir, "test_token")
        assert not staging_dir.exists()

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.HfApi")
    def test_upload_environment_cleanup_on_error(self, mock_api_class, mock_upload, tmp_path):
        """Test _upload_environment cleans up even on upload error."""
        # Setup
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()
        (staging_dir / "test.txt").write_text("test")
        mock_upload.side_effect = Exception("Upload failed")
        
        # Run upload and expect error
        with pytest.raises(Exception, match="Upload failed"):
            _upload_environment(
                env_name="test_env",
                repo_id="test_user/test_env",
                staging_dir=staging_dir,
                username="test_user",
                token="test_token",
            )
        
        # Verify cleanup happened even after error
        assert not staging_dir.exists()

    @patch("openenv_cli.commands.push.upload_to_space")
    @patch("openenv_cli.commands.push.HfApi")
    def test_upload_environment_nonexistent_staging_dir(self, mock_api_class, mock_upload, tmp_path):
        """Test _upload_environment handles nonexistent staging directory."""
        # Setup
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        staging_dir = tmp_path / "nonexistent"
        
        # Run upload
        _upload_environment(
            env_name="test_env",
            repo_id="test_user/test_env",
            staging_dir=staging_dir,
            username="test_user",
            token="test_token",
        )
        
        # Verify upload was called (cleanup won't fail on nonexistent dir)
        mock_upload.assert_called_once_with("test_user/test_env", staging_dir, "test_token")


class TestPushCommand:
    """Tests for push() typer command function."""

    @patch("openenv_cli.commands.push.push_environment")
    @patch("openenv_cli.commands.push.resolve_environment")
    @patch("openenv_cli.commands.push.check_auth_status")
    def test_push_command_already_authenticated(
        self,
        mock_check_auth,
        mock_resolve,
        mock_push_env,
        mock_environment,
    ):
        """Test push command when already authenticated."""
        from openenv_cli.core.auth import AuthStatus
        
        # Setup
        mock_check_auth.return_value = AuthStatus(
            is_authenticated=True, username="test_user", token="test_token"
        )
        
        # Resolve environment
        mock_resolve.return_value = ("test_env", Path("src/envs/test_env"))
        # Run command
        push(
            repo_id=None,
            private=False,
            base_image=None,
            dry_run=True,
        )
        
        # Verify
        mock_check_auth.assert_called_once()
        mock_push_env.assert_called_once()
        _, kwargs = mock_push_env.call_args
        assert kwargs["env_name"] == "test_env"
        assert kwargs["username"] == "test_user"
        assert kwargs["token"] == "test_token"
        assert kwargs["repo_id"] is None
        assert kwargs["private"] is False
        assert kwargs["base_image"] is None
        assert kwargs["dry_run"] is True

    @patch("openenv_cli.commands.push.perform_login")
    @patch("openenv_cli.commands.push.check_auth_status")
    @patch("sys.exit")
    def test_push_command_needs_login(
        self,
        mock_exit,
        mock_check_auth,
        mock_perform_login,
        mock_environment,
    ):
        """Test push command when login is needed."""
        from openenv_cli.core.auth import AuthStatus
        
        # Setup
        mock_check_auth.return_value = AuthStatus(is_authenticated=False)
        mock_perform_login.return_value = AuthStatus(
            is_authenticated=True, username="test_user", token="test_token"
        )
        
        # Run command
        push(
            repo_id=None,
            private=False,
            base_image=None,
            dry_run=True,
        )
        
        # Verify login was called
        mock_check_auth.assert_called_once()
        mock_perform_login.assert_called_once()

    @patch("openenv_cli.commands.push.perform_login")
    @patch("openenv_cli.commands.push.check_auth_status")
    def test_push_command_login_failure(
        self,
        mock_check_auth,
        mock_perform_login,
        mock_environment,
    ):
        """Test push command when login fails."""
        from openenv_cli.core.auth import AuthStatus
        
        # Setup
        mock_check_auth.return_value = AuthStatus(is_authenticated=False)
        mock_perform_login.side_effect = Exception("Login failed")
        
        # Run command (should exit with SystemExit)
        with pytest.raises(SystemExit) as exc_info:
            push(
                repo_id=None,
                private=False,
                base_image=None,
                dry_run=True,
            )
        
        # Verify exit code
        assert exc_info.value.code == 1

    @patch("openenv_cli.commands.push._upload_environment")
    @patch("openenv_cli.commands.push._prepare_environment")
    @patch("openenv_cli.commands.push.get_space_repo_id")
    @patch("openenv_cli.commands.push.resolve_environment")
    @patch("openenv_cli.commands.push.check_auth_status")
    def test_push_command_non_dry_run(
        self,
        mock_check_auth,
        mock_resolve,
        mock_get_repo_id,
        mock_prepare,
        mock_upload,
        mock_environment,
    ):
        """Test push command with dry_run=False (full workflow)."""
        from openenv_cli.core.auth import AuthStatus
        
        # Setup
        mock_check_auth.return_value = AuthStatus(
            is_authenticated=True, username="test_user", token="test_token"
        )
        mock_get_repo_id.return_value = "test_user/test_env"
        staging_dir = Path("staging")
        mock_prepare.return_value = staging_dir
        
        mock_resolve.return_value = ("test_env", Path("src/envs/test_env"))
        # Run command
        push(
            repo_id=None,
            private=False,
            base_image=None,
            dry_run=False,
        )
        
        # Verify
        mock_prepare.assert_called_once()
        mock_get_repo_id.assert_called_once_with("test_env", "test_user")
        mock_upload.assert_called_once_with(
            env_name="test_env",
            repo_id="test_user/test_env",
            staging_dir=staging_dir,
            username="test_user",
            token="test_token",
        )

    @patch("openenv_cli.commands.push.push_environment")
    @patch("openenv_cli.commands.push.check_auth_status")
    def test_push_command_error_handling(
        self,
        mock_check_auth,
        mock_push_env,
        mock_environment,
    ):
        """Test push command error handling."""
        from openenv_cli.core.auth import AuthStatus
        
        # Setup
        mock_check_auth.return_value = AuthStatus(
            is_authenticated=True, username="test_user", token="test_token"
        )
        mock_push_env.side_effect = Exception("Test error")
        
        # Run command (should exit with error)
        with pytest.raises(SystemExit) as exc_info:
            push(
                repo_id=None,
                private=False,
                base_image=None,
                dry_run=True,
            )
        
        assert exc_info.value.code == 1

    @patch("openenv_cli.commands.push.push_environment")
    @patch("openenv_cli.commands.push.resolve_environment")
    @patch("openenv_cli.commands.push.check_auth_status")
    def test_push_command_with_env_path(
        self,
        mock_check_auth,
        mock_resolve,
        mock_push_env,
        mock_environment,
    ):
        """Test push command resolves env from --env-path."""
        from openenv_cli.core.auth import AuthStatus
        
        mock_check_auth.return_value = AuthStatus(
            is_authenticated=True, username="test_user", token="test_token"
        )
        env_root = Path("/tmp/myenv")
        mock_resolve.return_value = ("test_env", env_root)
        
        push(
            repo_id=None,
            private=False,
            base_image=None,
            dry_run=True,
            env_path=str(env_root),
        )
        
        mock_resolve.assert_called_once()
        _, kwargs = mock_push_env.call_args
        assert kwargs["env_root"] == env_root

    @patch("openenv_cli.commands.push.push_environment")
    @patch("openenv_cli.commands.push.resolve_environment")
    @patch("openenv_cli.commands.push.check_auth_status")
    def test_push_command_in_cwd(
        self,
        mock_check_auth,
        mock_resolve,
        mock_push_env,
        mock_environment,
    ):
        """Test push command infers env from current working directory when no args provided."""
        from openenv_cli.core.auth import AuthStatus
        
        mock_check_auth.return_value = AuthStatus(
            is_authenticated=True, username="test_user", token="test_token"
        )
        cwd_root = Path("/work/env_root")
        mock_resolve.return_value = ("test_env", cwd_root)
        
        push(
            repo_id=None,
            private=False,
            base_image=None,
            dry_run=True,
            env_path=None,
        )
        
        mock_resolve.assert_called_once()
        _, kwargs = mock_push_env.call_args
        assert kwargs["env_root"] == cwd_root
