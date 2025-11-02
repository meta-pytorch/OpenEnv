# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Push command for deploying environments to HuggingFace Spaces."""

from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from ..core.auth import ensure_authenticated
from ..core.builder import (
    prepare_staging_directory,
    copy_environment_files,
    prepare_dockerfile,
    prepare_readme,
)
from ..core.space import space_exists, create_space, get_space_repo_id
from ..core.uploader import upload_to_space
from ..utils.env_loader import validate_environment


def push_environment(
    env_name: str,
    namespace: Optional[str] = None,
    space_name: Optional[str] = None,
    private: bool = False,
    base_image: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Push an environment to HuggingFace Spaces.
    
    Args:
        env_name: Name of the environment to push.
        namespace: Optional namespace (organization or user). If not provided,
                   uses the authenticated user's username.
        space_name: Optional custom space name. If not provided, uses env_name.
        private: Whether the space should be private (default: False).
        base_image: Base Docker image to use (default: ghcr.io/meta-pytorch/openenv-base:latest).
        dry_run: If True, prepare files but don't upload (default: False).
    """
    # Validate environment exists
    validate_environment(env_name)
    
    # Authenticate with HuggingFace
    username, token = ensure_authenticated()
    
    # Determine target space repo ID
    repo_id = get_space_repo_id(env_name, namespace=namespace, space_name=space_name)
    
    # Create HfApi instance
    api = HfApi(token=token)
    
    # Check if space exists, create if needed
    if not space_exists(api, repo_id):
        create_space(api, repo_id, private=private)
    else:
        print(f"Space {repo_id} already exists, will update it")
    
    # Set default base image if not provided
    if base_image is None:
        base_image = "ghcr.io/meta-pytorch/openenv-base:latest"
    
    # Prepare staging directory
    staging_dir = prepare_staging_directory(env_name, base_image)
    
    try:
        # Copy files
        copy_environment_files(env_name, staging_dir)
        
        # Prepare Dockerfile
        prepare_dockerfile(env_name, staging_dir, base_image)
        
        # Prepare README
        prepare_readme(env_name, staging_dir)
        
        if dry_run:
            print(f"Dry run: Files prepared in {staging_dir}")
            print(f"Would upload to: https://huggingface.co/spaces/{repo_id}")
        
        # Upload to space (skip if dry run)
        if not dry_run:
            print(f"Uploading to space: {repo_id}")
            upload_to_space(api, repo_id, staging_dir, token)
            print(f"✅ Successfully pushed {env_name} to https://huggingface.co/spaces/{repo_id}")
        
    finally:
        # Cleanup staging directory after upload or dry run
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir)
