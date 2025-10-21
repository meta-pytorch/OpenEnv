# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenSpiel Environment Integration.

This module provides integration between OpenSpiel games and the OpenEnv framework.
OpenSpiel (https://github.com/google-deepmind/open_spiel) is DeepMind's collection
of environments and algorithms for research in RL in games.

Supported games:
- Catch (1P)
- Tic-Tac-Toe (2P)
- Kuhn Poker (2P, imperfect info)
- Cliff Walking (1P)
- 2048 (1P)
- Blackjack (1P)
"""

from .client import OpenSpielEnv
from .models import OpenSpielAction, OpenSpielObservation, OpenSpielState

__all__ = ["OpenSpielEnv", "OpenSpielAction", "OpenSpielObservation", "OpenSpielState"]
