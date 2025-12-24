# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Atari Environment Server.

Server-side implementation of Atari environment for OpenEnv.
"""

from .atari_environment import AtariEnvironment

__all__ = ["AtariEnvironment"]
