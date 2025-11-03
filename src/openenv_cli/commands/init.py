"""Init command: scaffold a new OpenEnv environment from bundled templates."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from importlib import resources

from .._cli_utils import console


app = typer.Typer(help="Initialize a new OpenEnv environment")


def _copy_template_tree(template_pkg: str, template_dir: str, dest_dir: Path, env_name: str) -> None:
    """Copy the template tree from package resources into dest_dir, replacing placeholders."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    base = resources.files(template_pkg).joinpath(template_dir)

    def _iter(path: Path, rel: Path = Path("")) -> None:
        for entry in path.iterdir():
            rel_path = rel / entry.name
            # Replace placeholders in path components
            rel_str = str(rel_path).replace("__ENV_NAME__", env_name)
            target_path = dest_dir / rel_str
            if entry.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                _iter(entry, rel / entry.name)
            else:
                data = entry.read_bytes()
                try:
                    text = data.decode("utf-8")
                    text = text.replace("__ENV_NAME__", env_name)
                    target_path.write_text(text)
                except UnicodeDecodeError:
                    target_path.write_bytes(data)

    _iter(Path(base))


def _ensure_empty_or_force(target_dir: Path, force: bool) -> None:
    if target_dir.exists():
        if any(target_dir.iterdir()) and not force:
            raise FileExistsError(
                f"Target directory '{target_dir}' exists and is not empty. Use --force to overwrite."
            )


@app.command()
def init(
    env_name: Annotated[str, typer.Argument(help="Name of the environment to create")],
    path: Annotated[
        Path | None,
        typer.Option("--path", help="Parent directory to create the environment in (default: CWD)"),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite target directory if not empty")]=False,
) -> None:
    """Create a new OpenEnv environment from the bundled template."""
    target_parent = Path.cwd() if path is None else path
    target_dir = target_parent / env_name

    try:
        _ensure_empty_or_force(target_dir, force)
        # If directory exists and force, leave it; else create it
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy template
        _copy_template_tree(
            template_pkg="openenv_cli.templates",
            template_dir="openenv_env",
            dest_dir=target_dir,
            env_name=env_name,
        )

        # Git init
        subprocess.run(["git", "init"], cwd=str(target_dir), check=True)
        subprocess.run(["git", "add", "-A"], cwd=str(target_dir), check=True)
        subprocess.run(["git", "commit", "-m", "openenv: initial scaffold"], cwd=str(target_dir), check=True)

        console.print(f"[bold green]Created OpenEnv environment[/bold green]: {target_dir}")
        console.print("\nNext steps:")
        console.print(f"  1. cd {target_dir}")
        console.print(f"  2. uvicorn server.app:app --reload")
        console.print(f"  3. openenv push --repo-id <you>/{env_name}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise


