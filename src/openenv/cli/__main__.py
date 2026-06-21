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
from collections.abc import Callable

import typer
from openenv.cli.commands import (
    build,
    collect,
    fork,
    init,
    push,
    serve,
    skills,
    validate,
)

CommandCallback = Callable[..., object]
CommandSpec = tuple[str, str, CommandCallback]

# Create the main CLI app
app = typer.Typer(
    name="openenv",
    help="OpenEnv - An e2e framework for creating, deploying and using isolated execution environments for agentic RL training",
    no_args_is_help=True,
)

_COMMANDS_BEFORE_SKILLS: tuple[CommandSpec, ...] = (
    ("init", "Initialize a new OpenEnv environment", init.init),
    ("build", "Build Docker images for OpenEnv environments", build.build),
    (
        "validate",
        "Validate environment structure and deployment readiness",
        validate.validate,
    ),
    (
        "push",
        "Push an OpenEnv environment to Hugging Face Spaces or custom registry",
        push.push,
    ),
    ("serve", "Serve environments locally (TODO: Phase 4)", serve.serve),
    ("fork", "Fork (duplicate) a Hugging Face Space to your account", fork.fork),
)
_COMMANDS_AFTER_SKILLS: tuple[CommandSpec, ...] = (
    (
        "collect",
        "Collect rollouts from a deployed OpenEnv environment",
        collect.collect,
    ),
)


def _register_commands(commands: tuple[CommandSpec, ...]) -> None:
    """Register top-level Typer commands."""
    for name, help_text, callback in commands:
        app.command(name=name, help=help_text)(callback)


_register_commands(_COMMANDS_BEFORE_SKILLS)
app.add_typer(
    skills.app,
    name="skills",
    help="Manage OpenEnv skills for AI assistants",
)
_register_commands(_COMMANDS_AFTER_SKILLS)


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
