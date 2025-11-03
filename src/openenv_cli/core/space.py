# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Space management module for Hugging Face Spaces."""

from typing import Optional

from huggingface_hub import HfApi


def create_space(api: HfApi, repo_id: str, private: bool = False) -> None:
    """
    Create a Docker Space on Hugging Face.
    
    Args:
        api: HfApi instance to use for API calls.
        repo_id: Repository ID in format 'namespace/repo-name'.
        private: Whether the space should be private (default: False).
        
    Raises:
        Exception: If space creation fails.
    """
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=private,
            exist_ok=True  # Hub CLI handles existence checks and private/non-private re-creation
        )
    except Exception as e:
        # Check for authentication-related errors and provide clearer messages
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["unauthorized", "authentication", "401", "invalid token", "token"]):
            raise Exception(
                f"Authentication failed when creating space {repo_id}. "
                f"Please check your Hugging Face token and ensure it has write permissions. "
                f"Original error: {e}"
            ) from e
        
        # Check for permission-related errors
        if any(keyword in error_str for keyword in ["forbidden", "403", "permission", "not authorized"]):
            raise Exception(
                f"Permission denied when creating space {repo_id}. "
                f"Please verify you have permission to create spaces in this namespace. "
                f"Original error: {e}"
            ) from e
        
        # Raise a clearer generic error for other cases
        # Note: exist_ok=True handles space existence and will print warnings to the user
        raise Exception(
            f"Failed to create space {repo_id}: {e}"
        ) from e


def get_space_repo_id(env_name: str, username: str) -> str:
    """
    Get the full repository ID for a space using the provided username.
    
    Args:
        env_name: Environment name (e.g., "echo_env"). Used as space name.
        username: Username to use for the repository ID.
        
    Returns:
        Repository ID in format 'username/env_name'.
    """
    return f"{username}/{env_name}"
