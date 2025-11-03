# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Push command for deploying environments to Hugging Face Spaces."""

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from huggingface_hub import HfApi

from .._cli_utils import console, typer_factory
from ..core.auth import check_auth_status, perform_login
from ..core.builder import (
    prepare_staging_directory,
    copy_environment_files,
    prepare_dockerfile,
    prepare_readme,
)
from ..core.space import create_space, get_space_repo_id
from ..core.uploader import upload_to_space
from ..utils.env_loader import validate_environment


# Push command function (following HF Hub pattern for top-level commands like upload/download)
def push(
    env_name: Annotated[
        str,
        typer.Argument(help="Name of the environment to push (e.g., echo_env)"),
    ],
    repo_id: Annotated[
        Optional[str],
        typer.Option(
            "--repo-id",
            help="Hugging Face repository ID in format 'namespace/space-name'. "
            "If not provided, uses '{username}/{env_name}'.",
        ),
    ] = None,
    private: Annotated[
        bool,
        typer.Option("--private", help="Create a private space (default: public)"),
    ] = False,
    base_image: Annotated[
        Optional[str],
        typer.Option(
            "--base-image",
            help="Base Docker image to use (default: ghcr.io/meta-pytorch/openenv-base:latest)",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Prepare files but don't upload to Hugging Face"),
    ] = False,
) -> None:
    """
    Push an environment to Hugging Face Spaces.
    
    This command prepares and uploads an OpenEnv environment to Hugging Face Spaces,
    handling authentication, space creation, file preparation, and deployment.
    """
    try:
        # Handle authentication (all UI here)
        console.print("[bold cyan]Authenticating...[/bold cyan]", end=" ")
        auth_status = check_auth_status()
        
        if not auth_status.is_authenticated:
            # User needs to login - perform login (this will trigger prompts from huggingface_hub)
            # Note: login() prints ASCII art and prompts - this is expected behavior
            try:
                auth_status = perform_login()
                console.print(f"[bold green]✓ Authenticated as {auth_status.username}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] Login failed: {str(e)}")
                sys.exit(1)
        else:
            # Already authenticated
            console.print(f"[bold green]✓ Authenticated as {auth_status.username}[/bold green]")
        
        username, token = auth_status.get_credentials()
        
        # Print a newline to separate authentication from workflow messages
        console.print()
        
        if dry_run:
            status_message = f"[bold yellow]Preparing dry run for '{env_name}'...[/bold yellow]"
            with console.status(status_message):
                push_environment(
                    env_name=env_name,
                    username=username,
                    token=token,
                    repo_id=repo_id,
                    private=private,
                    base_image=base_image,
                    dry_run=dry_run,
                )
        else:
            # Use status spinner for preparation steps
            with console.status(f"[bold cyan]Preparing '{env_name}'...[/bold cyan]"):
                staging_dir = _prepare_environment(
                    env_name=env_name,
                    repo_id=repo_id,
                    private=private,
                    base_image=base_image,
                    username=username,
                    token=token,
                )
            
            # Determine repo_id for upload
            if repo_id is None:
                repo_id = get_space_repo_id(env_name, username)
            
            # Upload without spinner so messages from huggingface_hub appear cleanly
            _upload_environment(
                env_name=env_name,
                repo_id=repo_id,
                staging_dir=staging_dir,
                username=username,
                token=token,
            )

        if dry_run:
            console.print(
                f"[bold yellow]Dry run complete for '{env_name}'.[/bold yellow]"
            )
        else:
            console.print(
                f"[bold green]Successfully pushed '{env_name}'.[/bold green]"
            )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", highlight=False, soft_wrap=True)
        sys.exit(1)


def push_environment(
    env_name: str,
    username: str,
    token: str,
    repo_id: Optional[str] = None,
    private: bool = False,
    base_image: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Push an environment to Hugging Face Spaces.
    
    Args:
        env_name: Name of the environment to push.
        username: Authenticated username.
        token: Authentication token.
        repo_id: Optional repository ID in format 'namespace/space-name'. If not provided,
                 uses '{username}/{env_name}'.
        private: Whether the space should be private (default: False).
        base_image: Base Docker image to use (default: ghcr.io/meta-pytorch/openenv-base:latest).
        dry_run: If True, prepare files but don't upload (default: False).
    """
    # Validate environment exists
    validate_environment(env_name)
    
    # Determine target space repo ID
    if repo_id is None:
        repo_id = get_space_repo_id(env_name, username)
    
    # Create HfApi instance
    api = HfApi(token=token)
    
    # Check if space exists, create if needed
    create_space(api, repo_id, private=private)
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
        
        # Upload to space (skip if dry run)
        if not dry_run:
            upload_to_space(api, repo_id, staging_dir, token)
        
    finally:
        # Cleanup staging directory after upload or dry run
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir)


def _prepare_environment(
    env_name: str,
    repo_id: Optional[str],
    private: bool,
    base_image: Optional[str],
    username: str,
    token: str,
) -> Path:
    """
    Internal function to prepare environment staging directory.
    
    Returns:
        Path to staging directory (must be cleaned up by caller).
    """
    # Validate environment exists
    validate_environment(env_name)
    
    # Determine target space repo ID
    if repo_id is None:
        repo_id = get_space_repo_id(env_name, username)
    
    # Create HfApi instance
    api = HfApi(token=token)
    
    # Check if space exists, create if needed
    create_space(api, repo_id, private=private)
    
    # Set default base image if not provided
    if base_image is None:
        base_image = "ghcr.io/meta-pytorch/openenv-base:latest"
    
    # Prepare staging directory
    staging_dir = prepare_staging_directory(env_name, base_image)
    
    # Copy files
    copy_environment_files(env_name, staging_dir)
    
    # Prepare Dockerfile
    prepare_dockerfile(env_name, staging_dir, base_image)
    
    # Prepare README
    prepare_readme(env_name, staging_dir)
    
    return staging_dir


def _upload_environment(
    env_name: str,
    repo_id: str,
    staging_dir: Path,
    username: str,
    token: str,
) -> None:
    """
    Internal function to upload environment staging directory.
    
    The staging directory will be cleaned up after upload.
    """
    api = HfApi(token=token)
    
    try:
        upload_to_space(api, repo_id, staging_dir, token)
    finally:
        # Cleanup staging directory after upload
        if staging_dir.exists():
            import shutil
            shutil.rmtree(staging_dir)
