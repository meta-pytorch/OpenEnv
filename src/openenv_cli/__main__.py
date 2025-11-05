# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv CLI entry point.

This module provides the main entry point for the OpenEnv command-line interface,
following the Hugging Face CLI pattern.
"""

import sys

import typer

from openenv_cli.commands import init
from openenv_cli.commands import push

# Create the main CLI app
app = typer.Typer(
    name="openenv",
    help="OpenEnv - An e2e framework for creating, deploying and using isolated execution environments for agentic RL training",
    no_args_is_help=True,
)

# Register commands
app.command(name="init", help="Initialize a new OpenEnv environment")(init.init)
app.command(name="push", help="Push an OpenEnv environment to Hugging Face Spaces")(push.push)


# Entry point for setuptools
def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
