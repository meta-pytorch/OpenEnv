# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Authentication module for HuggingFace."""

import os
from typing import Optional, Tuple

from huggingface_hub import HfApi, login
from huggingface_hub.utils import get_token


def check_authentication() -> Optional[str]:
    """
    Check if user is authenticated with HuggingFace.
    
    Returns:
        Username if authenticated, None otherwise.
    """
    # Check for token in environment variable first
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        # Try to get token from stored credentials
        token = get_token()
    
    if not token:
        return None
    
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        return user_info.get("name")
    except Exception:
        # Invalid token or network error
        return None


def ensure_authenticated() -> Tuple[str, str]:
    """
    Ensure user is authenticated, prompting for login if needed.
    
    Returns:
        Tuple of (username, token).
        
    Raises:
        Exception: If authentication fails.
    """
    # Check for token in environment variable first
    token = os.environ.get("HF_TOKEN")
    
    if token:
        try:
            api = HfApi(token=token)
            user_info = api.whoami()
            return user_info.get("name"), token
        except Exception:
            pass  # Fall through to login
    
    # Check existing authentication
    username = check_authentication()
    if username:
        token = get_token() or os.environ.get("HF_TOKEN")
        if token:
            return username, token
    
    # Need to login
    username = login_interactive()
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise Exception("Failed to retrieve token after login")
    
    return username, token


def login_interactive() -> str:
    """
    Perform interactive login to HuggingFace.
    
    Returns:
        Username after successful login.
        
    Raises:
        Exception: If login fails.
    """
    try:
        login()
        # Verify login was successful
        username = check_authentication()
        if not username:
            raise Exception("Login failed: unable to verify authentication")
        return username
    except Exception as e:
        raise Exception(f"Login failed: {str(e)}")
