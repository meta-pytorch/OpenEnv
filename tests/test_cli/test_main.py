# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for __main__.py CLI entry point."""

import sys
from unittest.mock import patch

import pytest

from openenv_cli.__main__ import main


class TestMain:
    """Tests for main() function."""

    @patch("openenv_cli.__main__.app")
    def test_main_success(self, mock_app):
        """Test main() when app runs successfully."""
        mock_app.return_value = None
        
        # Should not raise
        main()
        
        mock_app.assert_called_once()

    @patch("openenv_cli.__main__.console")
    @patch("openenv_cli.__main__.app")
    @patch("sys.exit")
    def test_main_keyboard_interrupt(self, mock_exit, mock_app, mock_console):
        """Test main() handling KeyboardInterrupt."""
        mock_app.side_effect = KeyboardInterrupt()
        
        main()
        
        mock_console.print.assert_called_once()
        mock_exit.assert_called_once_with(130)

    @patch("openenv_cli.__main__.console")
    @patch("openenv_cli.__main__.app")
    @patch("sys.exit")
    def test_main_exception(self, mock_exit, mock_app, mock_console):
        """Test main() handling general Exception."""
        mock_app.side_effect = Exception("Test error")
        
        main()
        
        mock_console.print.assert_called_once()
        mock_exit.assert_called_once_with(1)

