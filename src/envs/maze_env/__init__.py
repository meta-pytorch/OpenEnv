# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Maze Environment Integration.

This module provides integration between Maze game and the OpenEnv framework.
"""

from .client import MazeEnv
from .models import MazeAction, MazeObservation, MazeState

__all__ = ["MazeEnv", "MazeAction", "MazeObservation", "MazeState"]
