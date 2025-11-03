# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for CLI utilities."""

import typer

from openenv_cli._cli_utils import typer_factory


class TestTyperFactory:
    """Tests for typer_factory function."""

    def test_typer_factory_returns_typer_app(self):
        """Test that typer_factory returns a Typer app."""
        app = typer_factory("Test help text")
        
        assert isinstance(app, typer.Typer)
        assert app.info.help == "Test help text"

    def test_typer_factory_creates_app_with_settings(self):
        """Test that typer_factory creates app with correct settings."""
        app = typer_factory("Test help")
        
        # Verify it's a Typer instance with expected configuration
        assert isinstance(app, typer.Typer)
        # The app should have the help text
        assert app.info.help == "Test help"

