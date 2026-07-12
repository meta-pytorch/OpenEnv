# SPDX-License-Identifier: BSD-3-Clause

"""
OpenEnv validate command.

This module provides the 'openenv validate' command to check if environments
are properly configured for multi-mode deployment.
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from openenv.validation import (
    format_shared_validation_report,
    RemoteValidationError,
    run_local_validation,
    run_remote_validation,
    ValidationProfile,
)


def _looks_like_url(value: str) -> bool:
    """Return True when the value appears to be a URL target."""
    candidate = value.strip().lower()
    return candidate.startswith("http://") or candidate.startswith("https://")


def validate(
    target: Annotated[
        str | None,
        typer.Argument(
            help=(
                "Path to the environment directory (default: current directory) "
                "or a running OpenEnv URL (http://... or https://...)"
            ),
        ),
    ] = None,
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help="Validate a running OpenEnv server by base URL (e.g. http://localhost:8000)",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output the RFC 008 shared validation report as JSON",
        ),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help="Validation profile: static, runtime, full, or publish",
        ),
    ] = None,
    remote: Annotated[
        bool,
        typer.Option(
            "--remote",
            help="Run validation in a fresh dedicated Hugging Face Sandbox",
        ),
    ] = False,
    hf_flavor: Annotated[
        str,
        typer.Option(
            "--hf-flavor",
            help="Hugging Face Sandbox hardware flavor used with --remote",
        ),
    ] = "cpu-basic",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Write the shared JSON validation report to this path",
        ),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help="HTTP timeout in seconds for runtime validation",
            min=0.1,
        ),
    ] = 5.0,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed information")
    ] = False,
) -> None:
    """
    Validate local environments and running OpenEnv servers.

    Local validation checks if an environment is properly configured with:
    - Required files (pyproject.toml, openenv.yaml, server/app.py, etc.)
    - Docker deployment support
    - uv run server capability
    - python -m module execution

    Runtime validation checks if a live OpenEnv server conforms to the
    versioned runtime API contract and returns a criteria-based JSON report.
    Every profile emits the RFC 008 shared report. The publish profile is a
    strict release gate: blocking skipped checks are incomplete and exit
    non-zero. `--remote` runs the same plan in a fresh dedicated Hugging Face
    Sandbox and returns the same report schema.
    Reports identify the served OpenEnv spec and pinned adapter; external task
    package formats are intentionally outside this command's dispatch surface.
    Automatic runtime launch is intended for trusted local source; connect to
    an already isolated server with `--url` for untrusted environments.

    Examples:

        ```bash
        # Validate current directory (recommended)
        $ cd my_env
        $ openenv validate

        # Validate a running environment and return JSON criteria
        $ openenv validate --url http://localhost:8000
        $ openenv validate https://my-env.hf.space

        # Validate with detailed output
        $ openenv validate --verbose

        # Validate specific environment
        $ openenv validate envs/echo_env

        # Run every locally available check and record remote-only skips
        $ openenv validate envs/echo_env --profile full --output report.json

        # Run the strict author gate remotely and save its structured guidance
        $ openenv validate envs/echo_env --profile publish --remote --output report.json
        ```
    """
    runtime_target = url
    if (
        runtime_target is not None
        and target is not None
        and not _looks_like_url(target)
    ):
        typer.echo(
            "Error: Cannot combine a local path argument with --url runtime validation",
            err=True,
        )
        raise typer.Exit(1)

    if target is not None and _looks_like_url(target):
        if runtime_target is not None and runtime_target != target:
            typer.echo(
                "Error: Conflicting runtime targets provided via argument and --url",
                err=True,
            )
            raise typer.Exit(1)
        runtime_target = target

    if remote and runtime_target is not None:
        typer.echo(
            "Error: --remote requires a local source directory, not a runtime URL",
            err=True,
        )
        raise typer.Exit(1)

    if profile is None:
        if remote:
            selected_profile = ValidationProfile.PUBLISH
        else:
            selected_profile = (
                ValidationProfile.RUNTIME
                if runtime_target is not None
                else ValidationProfile.STATIC
            )
    else:
        try:
            selected_profile = ValidationProfile(profile.lower())
        except ValueError as exc:
            typer.echo(
                "Error: --profile must be one of: static, runtime, full, publish",
                err=True,
            )
            raise typer.Exit(1) from exc

    if runtime_target is not None and selected_profile in {
        ValidationProfile.STATIC,
        ValidationProfile.PUBLISH,
    }:
        typer.echo(
            f"Error: The {selected_profile.value} profile requires a local source directory",
            err=True,
        )
        raise typer.Exit(1)

    if runtime_target is not None:
        shared_target: str | Path = runtime_target
        shared_runtime_url = runtime_target
    else:
        shared_target = Path.cwd() if target is None else Path(target)
        shared_runtime_url = None
        if not shared_target.exists():
            typer.echo(f"Error: Path does not exist: {shared_target}", err=True)
            raise typer.Exit(1)
        if not shared_target.is_dir():
            typer.echo(f"Error: Path is not a directory: {shared_target}", err=True)
            raise typer.Exit(1)

    try:
        if remote:
            assert isinstance(shared_target, Path)
            report = run_remote_validation(
                shared_target,
                profile=selected_profile,
                flavor=hf_flavor,
                runtime_timeout_s=timeout,
            )
        else:
            report = run_local_validation(
                shared_target,
                profile=selected_profile,
                runtime_url=shared_runtime_url,
                timeout_s=timeout,
            )
    except (RemoteValidationError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    serialized = json.dumps(report.to_dict(), indent=2)
    if output is not None:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"{serialized}\n", encoding="utf-8")
        except OSError as exc:
            typer.echo(
                f"Error: Unable to write validation report: {type(exc).__name__}",
                err=True,
            )
            raise typer.Exit(1) from exc

    # URL validation historically emitted JSON without an explicit --json;
    # retain that presentation while using the shared planner and schema.
    if json_output or (runtime_target is not None and profile is None):
        typer.echo(serialized)
    else:
        typer.echo(format_shared_validation_report(report, verbose=verbose))
        if output is not None:
            typer.echo(f"Report written to {output}")

    if not report.passed:
        raise typer.Exit(1)
