# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI entry point for OpenEnv."""

import sys

from ._cli_utils import console, typer_factory
from .commands.init import init as init_cmd
from .commands.push import push

# Create main typer app
app = typer_factory(help="OpenEnv CLI - Manage and deploy OpenEnv environments")

# Add callback to prevent single command from becoming implicit/default
@app.callback(invoke_without_command=True)
def main_callback() -> None:
    """OpenEnv CLI - Manage and deploy OpenEnv environments."""
    pass

# Register top-level commands (following HF Hub pattern for simple commands)
app.command(name="push", help="Push an environment to Hugging Face Spaces.")(push)
app.command(name="init", help="Initialize a new OpenEnv environment.")(init_cmd)


def main() -> None:
    """Main entry point for OpenEnv CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled.[/bold yellow]")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}", highlight=False, soft_wrap=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
