# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
connect4 Environment Server.

Server-side implementation of connect4 environment for OpenEnv.
"""

from .connect4_environment import Connect4Environment

__all__ = ["Connect4Environment"]
