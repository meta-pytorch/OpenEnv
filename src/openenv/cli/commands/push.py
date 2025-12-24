# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Push an OpenEnv environment to Hugging Face Spaces."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated
import sys
import typer
import yaml
from huggingface_hub import HfApi, login, whoami

from .._cli_utils import console, validate_env_structure

app = typer.Typer(help="Push an OpenEnv environment to Hugging Face Spaces")


def _validate_openenv_directory(directory: Path) -> tuple[str, dict]:
    """
    Validate that the directory is an OpenEnv environment.

    Returns:
        Tuple of (env_name, manifest_data)
    """
    # Use the comprehensive validation function
    try:
        warnings = validate_env_structure(directory)
        for warning in warnings:
            console.print(f"[bold yellow]⚠[/bold yellow] {warning}")
    except FileNotFoundError as e:
        raise typer.BadParameter(f"Invalid OpenEnv environment structure: {e}") from e

    # Load and validate manifest
    manifest_path = directory / "openenv.yaml"
    try:
        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)
    except Exception as e:
        raise typer.BadParameter(f"Failed to parse openenv.yaml: {e}") from e

    if not isinstance(manifest, dict):
        raise typer.BadParameter("openenv.yaml must be a YAML dictionary")

    env_name = manifest.get("name")
    if not env_name:
        raise typer.BadParameter("openenv.yaml must contain a 'name' field")

    return env_name, manifest


def _ensure_hf_authenticated() -> str:
    """
    Ensure user is authenticated with Hugging Face.

    Returns:
        Username of authenticated user
    """
    try:
        # Try to get current user
        user_info = whoami()
        # Handle both dict and object return types
        if isinstance(user_info, dict):
            username = (
                user_info.get("name")
                or user_info.get("fullname")
                or user_info.get("username")
            )
        else:
            # If it's an object, try to get name attribute
            username = (
                getattr(user_info, "name", None)
                or getattr(user_info, "fullname", None)
                or getattr(user_info, "username", None)
            )

        if not username:
            raise ValueError("Could not extract username from whoami response")

        console.print(f"[bold green]✓[/bold green] Authenticated as: {username}")
        return username
    except Exception:
        # Not authenticated, prompt for login
        console.print(
            "[bold yellow]Not authenticated with Hugging Face. Please login...[/bold yellow]"
        )

        try:
            login()
            # Verify login worked
            user_info = whoami()
            # Handle both dict and object return types
            if isinstance(user_info, dict):
                username = (
                    user_info.get("name")
                    or user_info.get("fullname")
                    or user_info.get("username")
                )
            else:
                username = (
                    getattr(user_info, "name", None)
                    or getattr(user_info, "fullname", None)
                    or getattr(user_info, "username", None)
                )

            if not username:
                raise ValueError("Could not extract username from whoami response")

            console.print(f"[bold green]✓[/bold green] Authenticated as: {username}")
            return username
        except Exception as e:
            raise typer.BadParameter(
                f"Hugging Face authentication failed: {e}. Please run login manually."
            ) from e


def _prepare_staging_directory(
    env_dir: Path,
    env_name: str,
    staging_dir: Path,
    base_image: str | None = None,
    enable_interface: bool = True,
) -> None:
    """
    Prepare files for deployment.

    This includes:
    - Copying necessary files
    - Modifying Dockerfile to optionally enable web interface and update base image
    - Ensuring README has proper HF frontmatter (if interface enabled)
    """
    # Create staging directory structure
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from env directory
    for item in env_dir.iterdir():
        # Skip hidden files and common ignore patterns
        if item.name.startswith(".") or item.name in ["__pycache__", ".git"]:
            continue

        dest = staging_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    # Ensure Dockerfile is at repository root (required by Hugging Face)
    dockerfile_server_path = staging_dir / "server" / "Dockerfile"
    dockerfile_root_path = staging_dir / "Dockerfile"
    dockerfile_path: Path | None = None

    if dockerfile_server_path.exists():
        if dockerfile_root_path.exists():
            dockerfile_root_path.unlink()
        dockerfile_server_path.rename(dockerfile_root_path)
        console.print(
            "[bold cyan]Moved Dockerfile to repository root for deployment[/bold cyan]"
        )
        dockerfile_path = dockerfile_root_path
    elif dockerfile_root_path.exists():
        dockerfile_path = dockerfile_root_path

    # Modify Dockerfile to optionally enable web interface and update base image
    if dockerfile_path and dockerfile_path.exists():
        dockerfile_content = dockerfile_path.read_text()
        lines = dockerfile_content.split("\n")
        new_lines = []
        cmd_found = False
        base_image_updated = False
        web_interface_env_exists = "ENABLE_WEB_INTERFACE" in dockerfile_content
        last_instruction = None

        for line in lines:
            stripped = line.strip()
            token = stripped.split(maxsplit=1)[0] if stripped else ""
            current_instruction = token.upper()

            is_healthcheck_continuation = last_instruction == "HEALTHCHECK"

            # Update base image if specified
            if base_image and stripped.startswith("FROM") and not base_image_updated:
                new_lines.append(f"FROM {base_image}")
                base_image_updated = True
                last_instruction = "FROM"
                continue

            if (
                stripped.startswith("CMD")
                and not cmd_found
                and not web_interface_env_exists
                and enable_interface
                and not is_healthcheck_continuation
            ):
                new_lines.append("ENV ENABLE_WEB_INTERFACE=true")
                cmd_found = True

            new_lines.append(line)

            if current_instruction:
                last_instruction = current_instruction

        if not cmd_found and not web_interface_env_exists and enable_interface:
            new_lines.append("ENV ENABLE_WEB_INTERFACE=true")

        if base_image and not base_image_updated:
            new_lines.insert(0, f"FROM {base_image}")

        dockerfile_path.write_text("\n".join(new_lines))

        changes = []
        if base_image and base_image_updated:
            changes.append("updated base image")
        if enable_interface and not web_interface_env_exists:
            changes.append("enabled web interface")
        if changes:
            console.print(
                f"[bold green]✓[/bold green] Updated Dockerfile: {', '.join(changes)}"
            )
    else:
        console.print(
            "[bold yellow]⚠[/bold yellow] No Dockerfile found at server/Dockerfile"
        )

    # Ensure README has proper HF frontmatter (only if interface enabled)
    if enable_interface:
        readme_path = staging_dir / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text()
            if "base_path: /web" not in readme_content:
                # Check if frontmatter exists
                if readme_content.startswith("---"):
                    # Add base_path to existing frontmatter
                    lines = readme_content.split("\n")
                    new_lines = []
                    _in_frontmatter = True
                    for i, line in enumerate(lines):
                        new_lines.append(line)
                        if line.strip() == "---" and i > 0:
                            # End of frontmatter, add base_path before this line
                            if "base_path:" not in "\n".join(new_lines):
                                new_lines.insert(-1, "base_path: /web")
                            _in_frontmatter = False
                    readme_path.write_text("\n".join(new_lines))
                else:
                    # No frontmatter, add it
                    frontmatter = f"""---
title: {env_name.replace("_", " ").title()} Environment Server
emoji: 🔊
colorFrom: '#00C9FF'
colorTo: '#1B2845'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

"""
                    readme_path.write_text(frontmatter + readme_content)
                console.print(
                    "[bold green]✓[/bold green] Updated README with HF Space frontmatter"
                )
        else:
            console.print("[bold yellow]⚠[/bold yellow] No README.md found")


def _create_hf_space(
    repo_id: str,
    api: HfApi,
    private: bool = False,
) -> None:
    """Create a Hugging Face Space if it doesn't exist."""
    console.print(f"[bold cyan]Creating/verifying space: {repo_id}[/bold cyan]")

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=private,
            exist_ok=True,
        )
        console.print(f"[bold green]✓[/bold green] Space {repo_id} is ready")
    except Exception as e:
        # Space might already exist, which is okay with exist_ok=True
        # But if there's another error, log it
        console.print(f"[bold yellow]⚠[/bold yellow] Space creation: {e}")


def _upload_to_hf_space(
    repo_id: str,
    staging_dir: Path,
    api: HfApi,
    private: bool = False,
) -> None:
    """Upload files to Hugging Face Space."""
    console.print(f"[bold cyan]Uploading files to {repo_id}...[/bold cyan]")

    try:
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[".git", "__pycache__", "*.pyc"],
        )
        console.print("[bold green]✓[/bold green] Upload completed successfully")
        console.print(
            f"[bold]Space URL:[/bold] https://huggingface.co/spaces/{repo_id}"
        )
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Upload failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def push(
    directory: Annotated[
        str | None,
        typer.Argument(
            help="Directory containing the OpenEnv environment (default: current directory)"
        ),
    ] = None,
    repo_id: Annotated[
        str | None,
        typer.Option(
            "--repo-id",
            "-r",
            help="Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)",
        ),
    ] = None,
    base_image: Annotated[
        str | None,
        typer.Option(
            "--base-image",
            "-b",
            help="Base Docker image to use (overrides Dockerfile FROM)",
        ),
    ] = None,
    interface: Annotated[
        bool,
        typer.Option(
            "--interface",
            help="Enable web interface (default: True if no registry specified)",
        ),
    ] = None,
    no_interface: Annotated[
        bool,
        typer.Option(
            "--no-interface",
            help="Disable web interface",
        ),
    ] = False,
    registry: Annotated[
        str | None,
        typer.Option(
            "--registry",
            help="Custom registry URL (e.g., docker.io/username). Disables web interface by default.",
        ),
    ] = None,
    private: Annotated[
        bool,
        typer.Option(
            "--private",
            help="Deploy the space as private",
        ),
    ] = False,
) -> None:
    """
    Push an OpenEnv environment to Hugging Face Spaces or a custom Docker registry.

    This command:
    1. Validates that the directory is an OpenEnv environment (openenv.yaml present)
    2. Builds and pushes to Hugging Face Spaces or custom Docker registry
    3. Optionally enables web interface for deployment

    The web interface is enabled by default when pushing to HuggingFace Spaces,
    but disabled by default when pushing to a custom Docker registry.

    Examples:
        # Push to HuggingFace Spaces from current directory (web interface enabled)
        $ cd my_env
        $ openenv push

        # Push to HuggingFace without web interface
        $ openenv push --no-interface

        # Push to Docker Hub
        $ openenv push --registry docker.io/myuser

        # Push to GitHub Container Registry
        $ openenv push --registry ghcr.io/myorg

        # Push to custom registry with web interface
        $ openenv push --registry myregistry.io/path1/path2 --interface

        # Push to specific HuggingFace repo
        $ openenv push --repo-id my-org/my-env

        # Push privately with custom base image
        $ openenv push --private --base-image ghcr.io/meta-pytorch/openenv-base:latest
    """
    # Handle interface flag logic
    if no_interface and interface:
        console.print(
            "[bold red]Error:[/bold red] Cannot specify both --interface and --no-interface",
            file=sys.stderr,
        )
        raise typer.Exit(1)

    # Determine if web interface should be enabled
    if no_interface:
        enable_interface = False
    elif interface is not None:
        enable_interface = interface
    elif registry is not None:
        # Custom registry: disable interface by default
        enable_interface = False
    else:
        # HuggingFace: enable interface by default
        enable_interface = True

    # Determine directory
    if directory:
        env_dir = Path(directory).resolve()
    else:
        env_dir = Path.cwd().resolve()

    if not env_dir.exists() or not env_dir.is_dir():
        raise typer.BadParameter(f"Directory does not exist: {env_dir}")

    # Check for openenv.yaml to confirm this is an environment directory
    openenv_yaml = env_dir / "openenv.yaml"
    if not openenv_yaml.exists():
        console.print(
            f"[bold red]Error:[/bold red] Not an OpenEnv environment directory (missing openenv.yaml): {env_dir}",
        )
        console.print(
            "[yellow]Hint:[/yellow] Run this command from the environment root directory",
        )
        raise typer.Exit(1)

    # Validate OpenEnv environment
    console.print(
        f"[bold cyan]Validating OpenEnv environment in {env_dir}...[/bold cyan]"
    )
    env_name, manifest = _validate_openenv_directory(env_dir)
    console.print(f"[bold green]✓[/bold green] Found OpenEnv environment: {env_name}")

    # Handle custom registry push
    if registry:
        console.print("[bold cyan]Preparing to push to custom registry...[/bold cyan]")
        if enable_interface:
            console.print("[bold cyan]Web interface will be enabled[/bold cyan]")

        # Import build functions
        from .build import _build_docker_image, _push_docker_image

        # Prepare build args for custom registry deployment
        build_args = {}
        if enable_interface:
            build_args["ENABLE_WEB_INTERFACE"] = "true"

        # Build Docker image from the environment directory
        tag = f"{registry}/{env_name}"
        console.print(f"[bold cyan]Building Docker image: {tag}[/bold cyan]")

        success = _build_docker_image(
            env_path=env_dir,
            tag=tag,
            build_args=build_args if build_args else None,
        )

        if not success:
            console.print("[bold red]✗ Docker build failed[/bold red]")
            raise typer.Exit(1)

        console.print("[bold green]✓ Docker build successful[/bold green]")

        # Push to registry
        console.print(f"[bold cyan]Pushing to registry: {registry}[/bold cyan]")

        success = _push_docker_image(
            tag, registry=None
        )  # Tag already includes registry

        if not success:
            console.print("[bold red]✗ Docker push failed[/bold red]")
            raise typer.Exit(1)

        console.print("\n[bold green]✓ Deployment complete![/bold green]")
        console.print(f"[bold]Image:[/bold] {tag}")
        return

    # Ensure authentication for HuggingFace
    username = _ensure_hf_authenticated()

    # Determine repo_id
    if not repo_id:
        repo_id = f"{username}/{env_name}"

    # Validate repo_id format
    if "/" not in repo_id or repo_id.count("/") != 1:
        raise typer.BadParameter(
            f"Invalid repo-id format: {repo_id}. Expected format: 'username/repo-name'"
        )

    # Initialize Hugging Face API
    api = HfApi()

    # Prepare staging directory
    deployment_type = (
        "with web interface" if enable_interface else "without web interface"
    )
    console.print(
        f"[bold cyan]Preparing files for Hugging Face deployment ({deployment_type})...[/bold cyan]"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir) / "staging"
        _prepare_staging_directory(
            env_dir,
            env_name,
            staging_dir,
            base_image=base_image,
            enable_interface=enable_interface,
        )

        # Create/verify space
        _create_hf_space(repo_id, api, private=private)

        # Upload files
        _upload_to_hf_space(repo_id, staging_dir, api, private=private)

        console.print("\n[bold green]✓ Deployment complete![/bold green]")
        console.print(f"Visit your space at: https://huggingface.co/spaces/{repo_id}")
