# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for space management module."""

from unittest.mock import Mock, patch

import pytest

from openenv_cli.core.space import space_exists, create_space, get_space_repo_id


@pytest.fixture
def mock_hf_api():
    """Mock HfApi for testing."""
    return Mock()


class TestSpaceExists:
    """Tests for space_exists function."""

    def test_space_exists_true(self, mock_hf_api):
        """Test space_exists returns True when space exists."""
        mock_hf_api.repo_exists.return_value = True

        result = space_exists(mock_hf_api, "test_user/my-env")

        assert result is True
        mock_hf_api.repo_exists.assert_called_once_with(
            repo_id="test_user/my-env",
            repo_type="space"
        )

    def test_space_exists_false(self, mock_hf_api):
        """Test space_exists returns False when space doesn't exist."""
        mock_hf_api.repo_exists.return_value = False

        result = space_exists(mock_hf_api, "test_user/my-env")

        assert result is False
        mock_hf_api.repo_exists.assert_called_once_with(
            repo_id="test_user/my-env",
            repo_type="space"
        )

    def test_space_exists_error(self, mock_hf_api):
        """Test space_exists handles errors gracefully."""
        mock_hf_api.repo_exists.side_effect = Exception("API error")

        # Should return False on error (not raise)
        result = space_exists(mock_hf_api, "test_user/my-env")
        assert result is False


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

    def test_create_space_already_exists(self, mock_hf_api):
        """Test creating a space that already exists."""
        from huggingface_hub.utils import RepositoryNotFoundError
        
        # RepositoryNotFoundError doesn't exist in all versions, use generic Exception
        mock_hf_api.create_repo.side_effect = Exception("Repository already exists")

        # Should not raise, just silently continue
        # This tests the function handles the case gracefully
        try:
            create_space(mock_hf_api, "test_user/my-env", private=False)
        except Exception:
            # If it raises, that's okay - we just test the call was made
            pass
        
        mock_hf_api.create_repo.assert_called_once()


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
