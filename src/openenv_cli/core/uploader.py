# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Uploader module for deploying to Hugging Face Spaces."""

from pathlib import Path

from huggingface_hub import upload_folder


def upload_to_space(
    repo_id: str,
    staging_dir: Path,
    token: str,
) -> None:
    """
    Upload staging directory contents to Hugging Face Space.
    
    Args:
        repo_id: Repository ID in format 'namespace/repo-name'.
        staging_dir: Path to staging directory to upload.
        token: Hugging Face token for authentication.
        
    Raises:
        Exception: If upload fails.
    """
    try:
        upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )
    except Exception as e:
        raise Exception(f"Failed to upload to space {repo_id}: {str(e)}")
