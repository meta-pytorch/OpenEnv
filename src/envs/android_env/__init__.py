# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Android Environment for OpenEnv.

This environment wraps DeepMind's android_env to provide RL agents with
access to Android applications and the Android operating system through
the OpenEnv framework.

The environment exposes Android devices as RL environments where agents
interact via touchscreen gestures and observe RGB pixel screens.
"""

from envs.android_env.client import AndroidEnv
from envs.android_env.models import AndroidAction, AndroidObservation

__all__ = ["AndroidEnv", "AndroidAction", "AndroidObservation"]
