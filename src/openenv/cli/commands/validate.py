# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv validate command.

This module provides the 'openenv validate' command to check if environments
are properly configured for multi-mode deployment.
"""

from pathlib import Path

import typer

from openenv.cli._validation import (
    format_validation_report,
    get_deployment_modes,
    validate_multi_mode_deployment,
)


def validate(
    env_path: str | None = typer.Argument(
        None, help="Path to the environment directory (default: current directory)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed information"
    ),
) -> None:
    """
    Validate an environment for standardized structure and deployment readiness.

    This command checks if an environment is properly configured with:
    - Required files (pyproject.toml, openenv.yaml, server/app.py, etc.)
    - Docker deployment support
    - uv run server capability
    - python -m module execution

    Examples:
        # Validate current directory (recommended)
        $ cd my_env
        $ openenv validate

        # Validate with detailed output
        $ openenv validate --verbose
        
        # Validate specific environment
        $ openenv validate envs/echo_env
    """
    # Determine environment path (default to current directory)
    if env_path is None:
        env_path_obj = Path.cwd()
    else:
        env_path_obj = Path(env_path)

    if not env_path_obj.exists():
        typer.echo(f"Error: Path does not exist: {env_path_obj}", err=True)
        raise typer.Exit(1)
    
    if not env_path_obj.is_dir():
        typer.echo(f"Error: Path is not a directory: {env_path_obj}", err=True)
        raise typer.Exit(1)
    
    # Check for openenv.yaml to confirm this is an environment directory
    openenv_yaml = env_path_obj / "openenv.yaml"
    if not openenv_yaml.exists():
        typer.echo(
            f"Error: Not an OpenEnv environment directory (missing openenv.yaml): {env_path_obj}",
            err=True,
        )
        typer.echo(
            "Hint: Run this command from the environment root directory or specify the path",
            err=True,
        )
        raise typer.Exit(1)

    env_name = env_path_obj.name
    if env_name.endswith("_env"):
        base_name = env_name[:-4]
    else:
        base_name = env_name

    # Run validation
    is_valid, issues = validate_multi_mode_deployment(env_path_obj)

    # Show validation report
    report = format_validation_report(base_name, is_valid, issues)
    typer.echo(report)

    # Show deployment modes if verbose
    if verbose:
        typer.echo("\nSupported deployment modes:")
        modes = get_deployment_modes(env_path_obj)
        for mode, supported in modes.items():
            status = "[YES]" if supported else "[NO]"
            typer.echo(f"  {status} {mode}")

        if is_valid:
            typer.echo("\nUsage examples:")
            typer.echo(f"  cd {env_path_obj.name} && uv run server")
            typer.echo(f"  cd {env_path_obj.name} && openenv build")
            typer.echo(f"  cd {env_path_obj.name} && openenv push")

    if not is_valid:
        raise typer.Exit(1)
