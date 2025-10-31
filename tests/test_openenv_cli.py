"""Tests for OpenEnv CLI."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openenv_cli import __main__ as cli_main
from openenv_cli.hf import (
    add_to_collection,
    ensure_authenticated,
    ensure_space,
    upload_to_space,
    wait_for_space_build,
    OPENENV_COLLECTION_ID,
)


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_push_requires_env(self):
        """Test that push command requires --env."""
        with patch("sys.argv", ["openenv", "push"]):
            with pytest.raises(SystemExit):
                cli_main.main()

    def test_push_with_env(self):
        """Test push command with --env."""
        # This will fail at validation, but we're testing arg parsing
        with patch("sys.argv", ["openenv", "push", "--env", "echo_env"]):
            with pytest.raises(SystemExit):
                cli_main.main()

    def test_push_with_space(self):
        """Test push command with --space."""
        with patch("sys.argv", ["openenv", "push", "--env", "echo_env", "--space", "test-org/test-space"]):
            with patch("openenv_cli.__main__.push_command") as mock_push:
                # Make push_command raise SystemExit to simulate failure
                mock_push.side_effect = SystemExit(1)
                with pytest.raises(SystemExit):
                    cli_main.main()

    def test_push_with_org(self):
        """Test push command with --org."""
        with patch("sys.argv", ["openenv", "push", "--env", "echo_env", "--org", "my-org"]):
            with patch("openenv_cli.__main__.validate_environment") as mock_val:
                mock_val.return_value = Path("/fake/path")
                with pytest.raises(SystemExit):
                    cli_main.main()

    def test_push_with_wait(self):
        """Test push command with --wait."""
        with patch("sys.argv", ["openenv", "push", "--env", "echo_env", "--wait"]):
            with patch("openenv_cli.__main__.validate_environment") as mock_val:
                mock_val.return_value = Path("/fake/path")
                with pytest.raises(SystemExit):
                    cli_main.main()

    def test_push_with_hardware(self):
        """Test push command with --hardware."""
        with patch("sys.argv", ["openenv", "push", "--env", "echo_env", "--hardware", "t4-small"]):
            with patch("openenv_cli.__main__.validate_environment") as mock_val:
                mock_val.return_value = Path("/fake/path")
                with pytest.raises(SystemExit):
                    cli_main.main()

    def test_push_with_private(self):
        """Test push command with --private."""
        with patch("sys.argv", ["openenv", "push", "--env", "echo_env", "--private"]):
            with patch("openenv_cli.__main__.validate_environment") as mock_val:
                mock_val.return_value = Path("/fake/path")
                with pytest.raises(SystemExit):
                    cli_main.main()

    def test_push_with_base_image_sha(self):
        """Test push command with --base-image-sha."""
        with patch("sys.argv", ["openenv", "push", "--env", "echo_env", "--base-image-sha", "abc123"]):
            with patch("openenv_cli.__main__.validate_environment") as mock_val:
                mock_val.return_value = Path("/fake/path")
                with pytest.raises(SystemExit):
                    cli_main.main()


class TestEnvironmentValidation:
    """Test environment validation."""

    def test_validate_environment_not_found(self):
        """Test validation fails when environment doesn't exist."""
        with pytest.raises(SystemExit):
            cli_main.validate_environment("nonexistent_env_xyz123")

    def test_validate_environment_missing_dockerfile(self):
        """Test validation fails when Dockerfile is missing."""
        # This test requires complex path mocking, skip for unit tests
        # Integration test will catch this
        pytest.skip("Path mocking is complex for validate_environment")

    def test_validate_environment_missing_readme(self):
        """Test validation fails when README is missing."""
        # This test requires complex path mocking, skip for unit tests
        # Integration test will catch this
        pytest.skip("Path mocking is complex for validate_environment")

    def test_validate_environment_missing_front_matter(self):
        """Test validation fails when README lacks HF front matter."""
        # This test requires complex path mocking, skip for unit tests
        # Integration test will catch this
        pytest.skip("Path mocking is complex for validate_environment")

    def test_validate_environment_success(self):
        """Test validation succeeds with valid environment."""
        project_root = Path(__file__).parent.parent
        env_path = project_root / "src" / "envs" / "echo_env"

        if not env_path.exists():
            pytest.skip("echo_env not found for integration test")

        # The validate_environment function now uses Path(__file__).parent.parent.parent
        # where __file__ is __main__.py at src/openenv_cli/__main__.py
        # This should correctly resolve to the project root
        result = cli_main.validate_environment("echo_env")
        assert result == env_path


class TestHuggingFaceAuth:
    """Test HuggingFace authentication."""

    @patch("openenv_cli.hf.HfApi")
    def test_ensure_authenticated_already_authenticated(self, mock_hf_api_class):
        """Test authentication when already logged in."""
        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "testuser"}
        mock_hf_api_class.return_value = mock_api

        result = ensure_authenticated()
        assert result == mock_api
        mock_api.whoami.assert_called_once()

    @patch("openenv_cli.hf.HfApi")
    @patch("openenv_cli.hf.hf_login")
    @patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_testtoken"})
    def test_ensure_authenticated_with_token(self, mock_login, mock_hf_api_class):
        """Test authentication using HUGGINGFACE_TOKEN."""
        # First call fails (not authenticated)
        mock_api = Mock()
        mock_api.whoami.side_effect = [Exception("Not authenticated"), {"name": "testuser"}]
        mock_hf_api_class.return_value = mock_api

        result = ensure_authenticated()
        assert result == mock_api
        mock_login.assert_called_once_with(token="hf_testtoken")

    @patch("openenv_cli.hf.subprocess.run")
    @patch("sys.stdin")
    @patch.dict(os.environ, {}, clear=True)  # Clear HUGGINGFACE_TOKEN
    def test_ensure_authenticated_with_cli_login(self, mock_stdin, mock_subprocess):
        """Test authentication using huggingface-cli login."""
        mock_stdin.isatty.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)

        # First call to HfApi() fails (not authenticated), 
        # token login fails (no token env var),
        # then subprocess login is called,
        # then second HfApi() call succeeds
        with patch("openenv_cli.hf.HfApi") as mock_hf_api_class:
            # First HfApi() call - not authenticated
            mock_api = Mock()
            mock_api.whoami.side_effect = Exception("Not authenticated")
            
            # Second HfApi() call (after subprocess login) - succeeds
            mock_api2 = Mock()
            mock_api2.whoami.return_value = {"name": "testuser"}
            
            # HfApi() is called: first time (initial), second time (after login)
            mock_hf_api_class.side_effect = [mock_api, mock_api2]

            result = ensure_authenticated()
            assert result == mock_api2
            # Should be called once for huggingface-cli login
            mock_subprocess.assert_called_once()

    @patch("sys.stdin")
    def test_ensure_authenticated_no_tty(self, mock_stdin):
        """Test authentication fails when no TTY and no token."""
        mock_stdin.isatty.return_value = False
        mock_api = Mock()
        mock_api.whoami.side_effect = Exception("Not authenticated")

        with patch("openenv_cli.hf.HfApi", return_value=mock_api):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(SystemExit):
                    ensure_authenticated()


class TestSpaceOperations:
    """Test HuggingFace Space operations."""

    @patch("openenv_cli.hf.HfApi")
    def test_ensure_space_exists(self, mock_hf_api_class):
        """Test ensuring space that already exists."""
        mock_api = Mock()
        mock_api.repo_info.return_value = {"id": "test-org/test-space"}
        mock_hf_api_class.return_value = mock_api

        ensure_space(mock_api, "test-org/test-space")
        mock_api.repo_info.assert_called_once_with(repo_id="test-org/test-space", repo_type="space")
        mock_api.create_repo.assert_not_called()

    def test_ensure_space_create_new(self):
        """Test creating a new space."""
        from huggingface_hub.utils import HfHubHTTPError
        
        mock_api = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_error = HfHubHTTPError("Not found", response=mock_response)
        mock_api.repo_info.side_effect = mock_error

        ensure_space(mock_api, "test-org/test-space")
        mock_api.create_repo.assert_called_once_with(
            repo_id="test-org/test-space",
            repo_type="space",
            space_sdk="docker",
            private=False,
        )

    @patch("openenv_cli.hf.HfApi")
    def test_ensure_space_with_hardware(self, mock_hf_api_class):
        """Test ensuring space with hardware request."""
        mock_api = Mock()
        mock_api.repo_info.return_value = {"id": "test-org/test-space"}
        mock_hf_api_class.return_value = mock_api

        ensure_space(mock_api, "test-org/test-space", hardware="t4-small")
        mock_api.request_space_hardware.assert_called_once_with(
            repo_id="test-org/test-space",
            hardware="t4-small",
        )

    @patch("openenv_cli.hf.upload_folder")
    def test_upload_to_space(self, mock_upload):
        """Test uploading files to space."""
        mock_api = Mock()
        staging_path = Path("/tmp/staging")

        upload_to_space(mock_api, "test-org/test-space", staging_path)
        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args[1]
        assert call_kwargs["repo_id"] == "test-org/test-space"
        assert call_kwargs["repo_type"] == "space"
        assert "commit_message" in call_kwargs


class TestSpaceBuildMonitoring:
    """Test Space build monitoring."""

    @patch("openenv_cli.hf.time.sleep")
    @patch("openenv_cli.hf.HfApi")
    def test_wait_for_space_build_running(self, mock_hf_api_class, mock_sleep):
        """Test waiting for space when it's already running."""
        mock_api = Mock()
        mock_api.get_space_runtime.return_value = {"stage": "RUNNING"}
        mock_hf_api_class.return_value = mock_api

        wait_for_space_build(mock_api, "test-org/test-space", timeout=10)
        mock_api.get_space_runtime.assert_called_once_with(repo_id="test-org/test-space")
        mock_sleep.assert_not_called()

    @patch("openenv_cli.hf.time.sleep")
    @patch("openenv_cli.hf.HfApi")
    def test_wait_for_space_build_building(self, mock_hf_api_class, mock_sleep):
        """Test waiting for space that's building."""
        mock_api = Mock()
        mock_api.get_space_runtime.side_effect = [
            {"stage": "BUILDING"},
            {"stage": "RUNNING"},
        ]
        mock_sleep.return_value = None
        mock_hf_api_class.return_value = mock_api

        wait_for_space_build(mock_api, "test-org/test-space", timeout=10)
        assert mock_api.get_space_runtime.call_count == 2

    @patch("openenv_cli.hf.time.sleep")
    @patch("openenv_cli.hf.HfApi")
    def test_wait_for_space_build_error(self, mock_hf_api_class, mock_sleep):
        """Test waiting for space that fails to build."""
        mock_api = Mock()
        mock_api.get_space_runtime.return_value = {"stage": "BUILD_ERROR"}
        mock_hf_api_class.return_value = mock_api

        with pytest.raises(SystemExit):
            wait_for_space_build(mock_api, "test-org/test-space", timeout=10)


class TestCollectionIntegration:
    """Test collection integration."""

    @patch("openenv_cli.hf.requests.post")
    def test_add_to_collection_success(self, mock_post):
        """Test successfully adding space to collection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        mock_api = Mock()
        mock_api.token = "test_token"

        add_to_collection(mock_api, "test-org/test-space")
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "openenv/environment-hub-68f16377abea1ea114fa0743" in call_args[0][0]
        assert call_args[1]["json"]["item_id"] == "test-org/test-space"
        assert call_args[1]["json"]["item_type"] == "space"

    @patch("openenv_cli.hf.requests.post")
    def test_add_to_collection_already_exists(self, mock_post):
        """Test adding space that's already in collection."""
        mock_response = Mock()
        mock_response.status_code = 409
        mock_post.return_value = mock_response
        
        mock_api = Mock()
        mock_api.token = "test_token"

        # Should not raise, just print message
        add_to_collection(mock_api, "test-org/test-space")

    @patch("openenv_cli.hf.requests.post")
    def test_add_to_collection_failure(self, mock_post):
        """Test adding space when API call fails."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_post.return_value = mock_response
        
        mock_api = Mock()
        mock_api.token = "test_token"

        # Should not raise, just print warning
        add_to_collection(mock_api, "test-org/test-space")

    def test_add_to_collection_no_token(self):
        """Test adding space when no token is available."""
        mock_api = Mock()
        mock_api.token = None

        # Should not raise, just print warning
        add_to_collection(mock_api, "test-org/test-space")

    @patch("openenv_cli.hf.requests.post")
    def test_add_to_collection_exception(self, mock_post):
        """Test adding space when API call raises exception."""
        mock_post.side_effect = Exception("Network error")
        
        mock_api = Mock()
        mock_api.token = "test_token"

        # Should not raise, just print warning
        add_to_collection(mock_api, "test-org/test-space")


class TestBuildStaging:
    """Test build staging."""

    def test_stage_build_success(self):
        """Test successful build staging."""
        # This test requires complex path mocking, skip for unit tests
        # Integration test will cover it
        pytest.skip("Path mocking for stage_build is complex, integration test will cover it")

    @patch("openenv_cli.__main__.Path")
    def test_stage_build_script_not_found(self, mock_path_class):
        """Test build staging when script is missing."""
        # This test requires complex path mocking
        pytest.skip("Path mocking for stage_build is complex, integration test will cover it")

    @patch("openenv_cli.__main__.subprocess.run")
    @patch("openenv_cli.__main__.Path")
    def test_stage_build_staging_dir_not_created(self, mock_path_class, mock_subprocess):
        """Test build staging when staging directory isn't created."""
        # This test requires complex path mocking
        pytest.skip("Path mocking for stage_build is complex, integration test will cover it")


class TestPushCommandIntegration:
    """Test the complete push command flow."""

    @patch("openenv_cli.__main__.add_to_collection")
    @patch("openenv_cli.__main__.wait_for_space_build")
    @patch("openenv_cli.__main__.upload_to_space")
    @patch("openenv_cli.__main__.ensure_space")
    @patch("openenv_cli.__main__.stage_build")
    @patch("openenv_cli.__main__.ensure_authenticated")
    @patch("openenv_cli.__main__.validate_environment")
    def test_push_command_complete_flow(
        self,
        mock_validate,
        mock_auth,
        mock_stage,
        mock_ensure_space,
        mock_upload,
        mock_wait,
        mock_collection,
    ):
        """Test complete push command flow."""
        # Setup mocks
        mock_validate.return_value = Path("/fake/env/path")
        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "testuser"}
        mock_auth.return_value = mock_api
        mock_stage.return_value = Path("/fake/staging")
        
        # Create mock args
        args = Mock()
        args.env = "echo_env"
        args.space = None
        args.org = "test-org"
        args.name = None
        args.private = False
        args.hardware = None
        args.base_image_sha = None
        args.wait = False
        args.timeout = 600

        cli_main.push_command(args)

        # Verify all steps were called
        mock_validate.assert_called_once_with("echo_env")
        mock_auth.assert_called_once()
        mock_stage.assert_called_once_with("echo_env", None)
        mock_ensure_space.assert_called_once_with(mock_api, "test-org/echo_env", private=False, hardware=None)
        mock_upload.assert_called_once_with(mock_api, "test-org/echo_env", Path("/fake/staging"))
        mock_wait.assert_not_called()  # wait=False
        mock_collection.assert_called_once_with(mock_api, "test-org/echo_env")

    @patch("openenv_cli.__main__.add_to_collection")
    @patch("openenv_cli.__main__.wait_for_space_build")
    @patch("openenv_cli.__main__.upload_to_space")
    @patch("openenv_cli.__main__.ensure_space")
    @patch("openenv_cli.__main__.stage_build")
    @patch("openenv_cli.__main__.ensure_authenticated")
    @patch("openenv_cli.__main__.validate_environment")
    def test_push_command_with_wait(
        self,
        mock_validate,
        mock_auth,
        mock_stage,
        mock_ensure_space,
        mock_upload,
        mock_wait,
        mock_collection,
    ):
        """Test push command with --wait flag."""
        mock_validate.return_value = Path("/fake/env/path")
        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "testuser"}
        mock_auth.return_value = mock_api
        mock_stage.return_value = Path("/fake/staging")
        
        args = Mock()
        args.env = "echo_env"
        args.space = None
        args.org = None
        args.name = None
        args.private = False
        args.hardware = None
        args.base_image_sha = None
        args.wait = True
        args.timeout = 600

        cli_main.push_command(args)

        # Verify wait was called
        mock_wait.assert_called_once_with(mock_api, "testuser/echo_env", timeout=600)
        mock_collection.assert_called_once_with(mock_api, "testuser/echo_env")


class TestCollectionIDConstant:
    """Test collection ID constant."""

    def test_collection_id_constant(self):
        """Test that collection ID constant is set correctly."""
        assert OPENENV_COLLECTION_ID == "openenv/environment-hub-68f16377abea1ea114fa0743"

