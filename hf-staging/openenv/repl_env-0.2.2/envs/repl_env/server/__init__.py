# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment Server Components.

This module contains the server-side implementation of the REPL environment.
"""

import sys
from pathlib import Path


def _prefer_bundled_openenv_src() -> None:
    """Ensure bundled src/openenv wins over installed openenv-core wheels."""
    for parent in Path(__file__).resolve().parents:
        src_dir = parent / "src"
        if not (src_dir / "openenv").is_dir():
            continue
        src_path = str(src_dir)
        if src_path in sys.path:
            sys.path.remove(src_path)
        sys.path.insert(0, src_path)
        return


_prefer_bundled_openenv_src()

from .python_executor import PythonExecutor
from .repl_environment import REPLEnvironment

__all__ = [
    "REPLEnvironment",
    "PythonExecutor",
]
