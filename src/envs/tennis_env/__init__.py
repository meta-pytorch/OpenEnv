# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tennis Environment for OpenEnv.

This package provides a Tennis game environment using the Arcade Learning Environment (ALE).
Supports single-agent training with reward shaping and symbolic state extraction.
"""

from .client import TennisEnv
from .models import TennisAction, TennisObservation, TennisState
from .server.tennis_environment import TennisEnvironment

__all__ = [
    "TennisEnv",
    "TennisAction",
    "TennisObservation",
    "TennisState",
    "TennisEnvironment",
]
