# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for uploader module."""

from unittest.mock import Mock, patch

import pytest

from openenv_cli.core.uploader import upload_to_space


@pytest.fixture
def mock_hf_api():
    """Mock HfApi for testing."""
    return Mock()


@pytest.fixture
def staging_dir(tmp_path):
    """Create a staging directory with test files."""
    staging = tmp_path / "staging"
    staging.mkdir()
    
    # Create some test files
    (staging / "Dockerfile").write_text("FROM test:latest")
    (staging / "README.md").write_text("# Test")
    (staging / "src" / "core").mkdir(parents=True)
    (staging / "src" / "core" / "__init__.py").write_text("# Core")
    
    return staging


class TestUploadToSpace:
    """Tests for upload_to_space function."""

    @patch("openenv_cli.core.uploader.upload_folder")
    def test_upload_to_space_uses_upload_folder(self, mock_upload, mock_hf_api, staging_dir):
        """Test that upload_to_space uses upload_folder for bulk upload."""
        mock_upload.return_value = None
        
        upload_to_space(mock_hf_api, "test_user/my-env", staging_dir, "test_token")
        
        mock_upload.assert_called_once()
        # Check that the call was made with correct parameters
        call_args = mock_upload.call_args
        assert call_args[1]["repo_id"] == "test_user/my-env"
        assert call_args[1]["folder_path"] == str(staging_dir)
        assert call_args[1]["repo_type"] == "space"
        assert call_args[1]["token"] == "test_token"

    @patch("openenv_cli.core.uploader.upload_folder")
    def test_upload_to_space_handles_errors(self, mock_upload, mock_hf_api, staging_dir):
        """Test that upload_to_space handles upload errors."""
        mock_upload.side_effect = Exception("Upload failed")
        
        with pytest.raises(Exception, match="Upload failed"):
            upload_to_space(mock_hf_api, "test_user/my-env", staging_dir, "test_token")

    @patch("openenv_cli.core.uploader.upload_folder")
    def test_upload_to_space_empty_directory(self, mock_upload, mock_hf_api, tmp_path):
        """Test uploading an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        mock_upload.return_value = None
        
        upload_to_space(mock_hf_api, "test_user/my-env", empty_dir, "test_token")
        
        mock_upload.assert_called_once()

    @patch("openenv_cli.core.uploader.upload_folder")
    def test_upload_to_space_nested_files(self, mock_upload, mock_hf_api, staging_dir):
        """Test uploading directory with nested files."""
        # Create nested structure
        (staging_dir / "src" / "envs" / "test_env").mkdir(parents=True)
        (staging_dir / "src" / "envs" / "test_env" / "models.py").write_text("# Models")
        
        mock_upload.return_value = None
        
        upload_to_space(mock_hf_api, "test_user/my-env", staging_dir, "test_token")
        
        mock_upload.assert_called_once()
