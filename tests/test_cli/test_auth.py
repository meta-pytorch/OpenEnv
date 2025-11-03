# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for authentication module."""

from unittest.mock import Mock, patch

import pytest

from openenv_cli.core.auth import (
    check_authentication,
    check_auth_status,
    ensure_authenticated,
    perform_login,
    AuthStatus,
)


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
    """Tests for ensure_authenticated function (legacy compatibility)."""

    @patch("openenv_cli.core.auth.check_auth_status")
    @patch("openenv_cli.core.auth.get_token")
    def test_ensure_authenticated_already_authenticated(self, mock_get_token, mock_check_status):
        """Test ensure_authenticated when already authenticated."""
        mock_check_status.return_value = AuthStatus(
            is_authenticated=True, username="test_user", token="test_token"
        )
        mock_get_token.return_value = "test_token"

        username, token = ensure_authenticated()

        assert username == "test_user"
        assert token == "test_token"
        mock_check_status.assert_called_once()

    @patch("openenv_cli.core.auth.perform_login")
    @patch("openenv_cli.core.auth.check_auth_status")
    @patch("openenv_cli.core.auth.get_token")
    def test_ensure_authenticated_needs_login(self, mock_get_token, mock_check_status, mock_perform_login):
        """Test ensure_authenticated when login is needed."""
        mock_check_status.return_value = AuthStatus(is_authenticated=False)
        mock_perform_login.return_value = AuthStatus(
            is_authenticated=True, username="new_user", token="new_token"
        )
        mock_get_token.return_value = "new_token"

        username, token = ensure_authenticated()

        assert username == "new_user"
        assert token == "new_token"
        mock_perform_login.assert_called_once()


class TestCheckAuthStatus:
    """Tests for check_auth_status function."""

    @patch("openenv_cli.core.auth.HfApi")
    @patch("openenv_cli.core.auth.get_token")
    def test_check_auth_status_authenticated(self, mock_get_token, mock_api_class):
        """Test check_auth_status when authenticated."""
        mock_get_token.return_value = "test_token"
        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test_user"}
        mock_api_class.return_value = mock_api

        status = check_auth_status()

        assert status.is_authenticated is True
        assert status.username == "test_user"
        assert status.token == "test_token"
        mock_api.whoami.assert_called_once()

    @patch("openenv_cli.core.auth.get_token")
    def test_check_auth_status_no_token(self, mock_get_token):
        """Test check_auth_status when no token exists."""
        mock_get_token.return_value = None

        status = check_auth_status()

        assert status.is_authenticated is False
        assert status.username is None
        assert status.token is None


class TestPerformLogin:
    """Tests for perform_login function."""

    @patch("openenv_cli.core.auth.check_auth_status")
    @patch("openenv_cli.core.auth.login")
    def test_perform_login_success(self, mock_login, mock_check_auth):
        """Test successful login."""
        mock_login.return_value = None  # login doesn't return anything
        mock_check_auth.return_value = AuthStatus(
            is_authenticated=True, username="logged_in_user", token="new_token"
        )

        status = perform_login()

        assert status.is_authenticated is True
        assert status.username == "logged_in_user"
        mock_login.assert_called_once()
        mock_check_auth.assert_called_once()

    @patch("openenv_cli.core.auth.login")
    def test_perform_login_failure(self, mock_login):
        """Test login failure."""
        mock_login.side_effect = Exception("Login failed")
        
        with pytest.raises(Exception, match="Login failed"):
            perform_login()

    @patch("openenv_cli.core.auth.check_auth_status")
    @patch("openenv_cli.core.auth.login")
    def test_perform_login_verification_fails(self, mock_login, mock_check_auth):
        """Test login when verification fails."""
        mock_login.return_value = None
        mock_check_auth.return_value = AuthStatus(is_authenticated=False)
        
        with pytest.raises(Exception, match="Login failed: unable to verify"):
            perform_login()

    @patch("openenv_cli.core.auth.check_auth_status")
    @patch("openenv_cli.core.auth.login")
    def test_perform_login_verification_exception_wrapped(self, mock_login, mock_check_auth):
        """Test login when verification raises exception."""
        mock_login.side_effect = ValueError("Network error")
        
        with pytest.raises(Exception, match="Login failed: Network error"):
            perform_login()

    @patch("openenv_cli.core.auth.HfApi")
    @patch("openenv_cli.core.auth.get_token")
    def test_check_auth_status_no_username(self, mock_get_token, mock_api_class):
        """Test check_auth_status when whoami returns no username."""
        mock_get_token.return_value = "test_token"
        mock_api = Mock()
        mock_api.whoami.return_value = {}  # No "name" key
        mock_api_class.return_value = mock_api
        
        status = check_auth_status()
        
        assert status.is_authenticated is False
        assert status.username is None

    @patch("openenv_cli.core.auth.HfApi")
    @patch("openenv_cli.core.auth.get_token")
    def test_check_auth_status_api_exception(self, mock_get_token, mock_api_class):
        """Test check_auth_status when API call raises exception."""
        mock_get_token.return_value = "test_token"
        mock_api = Mock()
        mock_api.whoami.side_effect = Exception("API error")
        mock_api_class.return_value = mock_api
        
        status = check_auth_status()
        
        assert status.is_authenticated is False


class TestAuthStatus:
    """Tests for AuthStatus dataclass."""

    def test_get_credentials_success(self):
        """Test get_credentials when authenticated."""
        status = AuthStatus(
            is_authenticated=True, username="test_user", token="test_token"
        )
        
        username, token = status.get_credentials()
        
        assert username == "test_user"
        assert token == "test_token"

    def test_get_credentials_not_authenticated(self):
        """Test get_credentials when not authenticated."""
        status = AuthStatus(is_authenticated=False)
        
        with pytest.raises(Exception, match="Not authenticated"):
            status.get_credentials()

    def test_get_credentials_missing_username(self):
        """Test get_credentials when username is missing."""
        status = AuthStatus(is_authenticated=True, username=None, token="test_token")
        
        with pytest.raises(Exception, match="Not authenticated"):
            status.get_credentials()

    def test_get_credentials_missing_token(self):
        """Test get_credentials when token is missing."""
        status = AuthStatus(is_authenticated=True, username="test_user", token=None)
        
        with pytest.raises(Exception, match="Not authenticated"):
            status.get_credentials()
