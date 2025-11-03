# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CLI utilities and helpers."""

import typer
from rich.console import Console
from rich.traceback import install

# Initialize console and install rich traceback handler
console = Console()
install(show_locals=False)


def typer_factory(help: str) -> typer.Typer:
    """
    Create a Typer app with consistent settings.
    
    Args:
        help: Help text for the app.
        
    Returns:
        Configured Typer app instance.
    """
    return typer.Typer(
        help=help,
        add_completion=True,
        no_args_is_help=True,
        # Use rich for better formatting
        rich_markup_mode="rich",
        pretty_exceptions_show_locals=False,
    )

