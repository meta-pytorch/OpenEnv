# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Authentication module for Hugging Face."""

from dataclasses import dataclass
from typing import Optional, Tuple

from huggingface_hub import HfApi, login
from huggingface_hub.utils import get_token


@dataclass
class AuthStatus:
    """Authentication status."""
    is_authenticated: bool
    username: Optional[str] = None
    token: Optional[str] = None
    
    def get_credentials(self) -> Tuple[str, str]:
        """Get username and token, raising if not authenticated."""
        if not self.is_authenticated or not self.username or not self.token:
            raise Exception("Not authenticated")
        return self.username, self.token


def check_auth_status() -> AuthStatus:
    """
    Check if user is authenticated with Hugging Face.
    
    Returns:
        AuthStatus object indicating authentication state.
    """
    # Get token from stored credentials (from previous login)
    token = get_token()
    
    if not token:
        return AuthStatus(is_authenticated=False)
    
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info.get("name")
        if username:
            return AuthStatus(is_authenticated=True, username=username, token=token)
        return AuthStatus(is_authenticated=False)
    except Exception:
        # Invalid token or network error
        return AuthStatus(is_authenticated=False)


def perform_login() -> AuthStatus:
    """
    Perform interactive login to Hugging Face.
    
    Note: This function will trigger interactive prompts (handled by huggingface_hub.login()).
    The caller should handle UI/prompts appropriately.
    
    Returns:
        AuthStatus after login attempt.
        
    Raises:
        Exception: If login fails.
    """
    try:
        login()
        # Verify login was successful
        status = check_auth_status()
        if not status.is_authenticated:
            raise Exception("Login failed: unable to verify authentication")
        return status
    except Exception as e:
        # Re-raise our custom exceptions, wrap others
        if isinstance(e, Exception) and str(e).startswith("Login failed"):
            raise
        raise Exception(f"Login failed: {str(e)}")


# Legacy functions for backward compatibility during refactoring
def check_authentication() -> Optional[str]:
    """
    Check if user is authenticated with Hugging Face.
    
    Returns:
        Username if authenticated, None otherwise.
    """
    status = check_auth_status()
    return status.username if status.is_authenticated else None


def ensure_authenticated() -> Tuple[str, str]:
    """
    Ensure user is authenticated, prompting for login if needed.
    
    Note: This function triggers interactive prompts. For better separation of concerns,
    consider using check_auth_status() and perform_login() separately.
    
    Returns:
        Tuple of (username, token).
        
    Raises:
        Exception: If authentication fails.
    """
    status = check_auth_status()
    if status.is_authenticated:
        return status.get_credentials()
    
    # Need to login
    status = perform_login()
    return status.get_credentials()
