# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Space management module for HuggingFace Spaces."""

from typing import Optional

from huggingface_hub import HfApi

from .auth import ensure_authenticated


def space_exists(api: HfApi, repo_id: str) -> bool:
    """
    Check if a Docker Space exists.
    
    Args:
        api: HfApi instance to use for API calls.
        repo_id: Repository ID in format 'namespace/repo-name'.
        
    Returns:
        True if space exists, False otherwise.
    """
    try:
        return api.repo_exists(repo_id=repo_id, repo_type="space")
    except Exception:
        return False


def create_space(api: HfApi, repo_id: str, private: bool = False) -> None:
    """
    Create a Docker Space on HuggingFace.
    
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
            exist_ok=True  # Don't fail if space already exists
        )
    except Exception as e:
        # If space already exists, that's okay
        if "already exists" in str(e).lower() or "repository already exists" in str(e).lower():
            return
        raise


def get_space_repo_id(env_name: str, namespace: Optional[str] = None, space_name: Optional[str] = None) -> str:
    """
    Get the full repository ID for a space.
    
    Args:
        env_name: Environment name (e.g., "echo_env"). Used as space name if space_name is not provided.
        namespace: Optional namespace (organization or user). If not provided,
                   uses the authenticated user's username.
        space_name: Optional custom space name. If provided, used instead of env_name.
        
    Returns:
        Repository ID in format 'namespace/space-name'.
    """
    # Use space_name if provided, otherwise use env_name
    repo_name = space_name if space_name else env_name
    
    if namespace:
        return f"{namespace}/{repo_name}"
    
    # Use authenticated user's username
    username, _ = ensure_authenticated()
    return f"{username}/{repo_name}"
