# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Wordle environment.
"""

from core.env_server import create_fastapi_app

from ..models import WordleAction, WordleObservation
from .wordle_environment import WordleEnvironment

# Create environment instance
env = WordleEnvironment(max_attempts=6)

# Create FastAPI app
app = create_fastapi_app(env, WordleAction, WordleObservation)
