# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for space management module."""

from unittest.mock import Mock, patch

import pytest

from openenv_cli.core.space import create_space, get_space_repo_id


@pytest.fixture
def mock_hf_api():
    """Mock HfApi for testing."""
    return Mock()


class TestCreateSpace:
    """Tests for create_space function."""

    def test_create_space_public(self, mock_hf_api):
        """Test creating a public space."""
        mock_hf_api.create_repo.return_value = None

        create_space(mock_hf_api, "test_user/my-env", private=False)

        mock_hf_api.create_repo.assert_called_once_with(
            repo_id="test_user/my-env",
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True
        )

    def test_create_space_private(self, mock_hf_api):
        """Test creating a private space."""
        mock_hf_api.create_repo.return_value = None

        create_space(mock_hf_api, "test_user/my-env", private=True)

        mock_hf_api.create_repo.assert_called_once_with(
            repo_id="test_user/my-env",
            repo_type="space",
            space_sdk="docker",
            private=True,
            exist_ok=True
        )


    def test_create_space_authentication_error(self, mock_hf_api):
        """Test that authentication errors are raised with clearer messages."""
        mock_hf_api.create_repo.side_effect = Exception("401 Unauthorized: Invalid token")

        with pytest.raises(Exception) as exc_info:
            create_space(mock_hf_api, "test_user/my-env", private=False)
        
        error_message = str(exc_info.value)
        assert "Authentication failed" in error_message
        assert "test_user/my-env" in error_message
        assert "Hugging Face token" in error_message
        assert "write permissions" in error_message

    def test_create_space_permission_error(self, mock_hf_api):
        """Test that permission errors are raised with clearer messages."""
        mock_hf_api.create_repo.side_effect = Exception("403 Forbidden: Not authorized")

        with pytest.raises(Exception) as exc_info:
            create_space(mock_hf_api, "test_user/my-env", private=False)
        
        error_message = str(exc_info.value)
        assert "Permission denied" in error_message
        assert "test_user/my-env" in error_message
        assert "permission to create spaces" in error_message

    def test_create_space_generic_error(self, mock_hf_api):
        """Test that generic errors are raised with clearer messages."""
        mock_hf_api.create_repo.side_effect = Exception("Network error occurred")

        with pytest.raises(Exception) as exc_info:
            create_space(mock_hf_api, "test_user/my-env", private=False)
        
        error_message = str(exc_info.value)
        assert "Failed to create space" in error_message
        assert "test_user/my-env" in error_message
        assert "Network error occurred" in error_message


class TestGetSpaceRepoId:
    """Tests for get_space_repo_id function."""

    @patch("openenv_cli.core.space.ensure_authenticated")
    def test_get_space_repo_id_with_namespace(self, mock_auth):
        """Test get_space_repo_id with explicit namespace."""
        mock_auth.return_value = ("user", "token")

        result = get_space_repo_id("my-env", namespace="my-org")

        assert result == "my-org/my-env"

    @patch("openenv_cli.core.space.ensure_authenticated")
    def test_get_space_repo_id_no_namespace(self, mock_auth):
        """Test get_space_repo_id without namespace uses authenticated user."""
        mock_auth.return_value = ("test_user", "token")

        result = get_space_repo_id("my-env")

        assert result == "test_user/my-env"

    @patch("openenv_cli.core.space.ensure_authenticated")
    def test_get_space_repo_id_env_name_with_underscore(self, mock_auth):
        """Test get_space_repo_id handles env names with underscores."""
        mock_auth.return_value = ("test_user", "token")

        result = get_space_repo_id("my_env", namespace="my-org")

        assert result == "my-org/my_env"

    @patch("openenv_cli.core.space.ensure_authenticated")
    def test_get_space_repo_id_with_space_name(self, mock_auth):
        """Test get_space_repo_id with custom space name."""
        mock_auth.return_value = ("test_user", "token")

        result = get_space_repo_id("my_env", space_name="custom-space")

        assert result == "test_user/custom-space"

    @patch("openenv_cli.core.space.ensure_authenticated")
    def test_get_space_repo_id_with_namespace_and_space_name(self, mock_auth):
        """Test get_space_repo_id with both namespace and space name."""
        result = get_space_repo_id("my_env", namespace="my-org", space_name="custom-space")

        assert result == "my-org/custom-space"
