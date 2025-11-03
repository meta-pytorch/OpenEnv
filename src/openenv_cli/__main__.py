# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI entry point for OpenEnv."""

import argparse
import sys

from rich.console import Console
from rich.traceback import install

from .commands.push import push_environment
from .core.auth import ensure_authenticated


console = Console()
install(show_locals=False)


def main():
    """Main entry point for OpenEnv CLI."""
    parser = argparse.ArgumentParser(
        prog="openenv",
        description="OpenEnv CLI - Manage and deploy OpenEnv environments",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Push command
    push_parser = subparsers.add_parser(
        "push",
        help="Push an environment to Hugging Face Spaces",
    )
    push_parser.add_argument(
        "env_name",
        help="Name of the environment to push (e.g., echo_env)",
    )
    push_parser.add_argument(
        "--repo-id",
        help="Hugging Face repository ID in format 'namespace/space-name'. "
             "If not provided, uses '{username}/{env_name}'.",
    )
    push_parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private space (default: public)",
    )
    push_parser.add_argument(
        "--base-image",
        help="Base Docker image to use "
             "(default: ghcr.io/meta-pytorch/openenv-base:latest)",
    )
    push_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files but don't upload to Hugging Face",
    )
    
    args = parser.parse_args()
    
    if args.command == "push":
        try:
            # Authenticate first (before status spinner) to allow interactive login if needed
            # Note: login() may print ASCII art to stdout - we clean up after
            username, token = ensure_authenticated()
            # Print a newline to separate login output from our status message
            console.print()  # Clean separator after login output
            
            if args.dry_run:
                status_message = f"[bold yellow]Preparing dry run for '{args.env_name}'...[/bold yellow]"
                with console.status(status_message):
                    push_environment(
                        env_name=args.env_name,
                        repo_id=args.repo_id,
                        private=args.private,
                        base_image=args.base_image,
                        dry_run=args.dry_run,
                    )
            else:
                # Use status spinner for preparation steps
                with console.status(f"[bold cyan]Preparing '{args.env_name}'...[/bold cyan]"):
                    from openenv_cli.commands.push import _prepare_environment
                    staging_dir = _prepare_environment(
                        env_name=args.env_name,
                        repo_id=args.repo_id,
                        private=args.private,
                        base_image=args.base_image,
                        username=username,
                        token=token,
                    )
                
                # Determine repo_id for upload
                if args.repo_id is None:
                    from openenv_cli.core.space import get_space_repo_id
                    repo_id = get_space_repo_id(args.env_name)
                else:
                    repo_id = args.repo_id
                
                # Upload without spinner so messages from huggingface_hub appear cleanly
                from openenv_cli.commands.push import _upload_environment
                _upload_environment(
                    env_name=args.env_name,
                    repo_id=repo_id,
                    staging_dir=staging_dir,
                    username=username,
                    token=token,
                )

            if args.dry_run:
                console.print(
                    f"[bold yellow]Dry run complete for '{args.env_name}'.[/bold yellow]"
                )
            else:
                console.print(
                    f"[bold green]Successfully pushed '{args.env_name}'.[/bold green]"
                )
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}", highlight=False, soft_wrap=True)
            sys.exit(1)
    else:
        console.print(parser.format_help())
        sys.exit(1)


if __name__ == "__main__":
    main()
