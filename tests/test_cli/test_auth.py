# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for authentication module."""

from unittest.mock import Mock, patch

import pytest

from openenv_cli.core.auth import check_authentication, ensure_authenticated, login_interactive


@pytest.fixture
def mock_hf_api():
    """Mock HfApi for testing."""
    return Mock()


@pytest.fixture
def mock_get_token():
    """Mock get_token for testing."""
    return Mock()


class TestCheckAuthentication:
    """Tests for check_authentication function."""

    @patch("openenv_cli.core.auth.HfApi")
    @patch("openenv_cli.core.auth.get_token")
    def test_check_authentication_with_valid_token(self, mock_get_token, mock_api_class):
        """Test check_authentication with valid token."""
        mock_get_token.return_value = "test_token"
        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test_user"}
        mock_api_class.return_value = mock_api

        result = check_authentication()

        assert result == "test_user"
        mock_api.whoami.assert_called_once()

    @patch("openenv_cli.core.auth.HfApi")
    @patch("openenv_cli.core.auth.get_token")
    def test_check_authentication_no_token(self, mock_get_token, mock_api_class):
        """Test check_authentication when no token exists."""
        mock_get_token.return_value = None
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        result = check_authentication()

        assert result is None

    @patch("openenv_cli.core.auth.HfApi")
    @patch("openenv_cli.core.auth.get_token")
    def test_check_authentication_invalid_token(self, mock_get_token, mock_api_class):
        """Test check_authentication with invalid token."""
        mock_get_token.return_value = "invalid_token"
        mock_api = Mock()
        mock_api.whoami.side_effect = Exception("Invalid token")
        mock_api_class.return_value = mock_api

        result = check_authentication()

        assert result is None


class TestEnsureAuthenticated:
    """Tests for ensure_authenticated function."""

    @patch("openenv_cli.core.auth.check_authentication")
    @patch("openenv_cli.core.auth.get_token")
    def test_ensure_authenticated_already_authenticated(self, mock_get_token, mock_check):
        """Test ensure_authenticated when already authenticated."""
        mock_check.return_value = "test_user"
        mock_get_token.return_value = "test_token"

        username, token = ensure_authenticated()

        assert username == "test_user"
        assert token == "test_token"
        mock_check.assert_called_once()

    @patch("openenv_cli.core.auth.check_authentication")
    @patch("openenv_cli.core.auth.login_interactive")
    @patch("openenv_cli.core.auth.get_token")
    def test_ensure_authenticated_needs_login(self, mock_get_token, mock_login, mock_check):
        """Test ensure_authenticated when login is needed."""
        mock_check.return_value = None
        mock_login.return_value = "new_user"
        mock_get_token.return_value = "new_token"

        username, token = ensure_authenticated()

        assert username == "new_user"
        assert token == "new_token"
        mock_login.assert_called_once()


class TestLoginInteractive:
    """Tests for login_interactive function."""

    @patch("openenv_cli.core.auth.login")
    @patch("openenv_cli.core.auth.HfApi")
    @patch("openenv_cli.core.auth.get_token")
    def test_login_interactive_success(self, mock_get_token, mock_api_class, mock_login):
        """Test successful interactive login."""
        mock_login.return_value = None  # login doesn't return anything
        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "logged_in_user"}
        mock_api_class.return_value = mock_api
        mock_get_token.return_value = "new_token"

        username = login_interactive()

        assert username == "logged_in_user"
        mock_login.assert_called_once()

    @patch("openenv_cli.core.auth.login")
    def test_login_interactive_failure(self, mock_login):
        """Test interactive login failure."""
        mock_login.side_effect = Exception("Login failed")

        with pytest.raises(Exception, match="Login failed"):
            login_interactive()
