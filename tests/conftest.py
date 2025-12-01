# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pytest configuration for OpenEnv tests.

This file adds the src directory to sys.path so that tests can import
core, envs, and other modules from the src directory.

NOTE: Do not create __init__.py files in test directories that have
the same name as source directories (e.g., tests/core/) to avoid
import conflicts.
"""

import sys
from pathlib import Path

# Add src to path for tests to find core and envs modules
_src_path = str(Path(__file__).resolve().parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
