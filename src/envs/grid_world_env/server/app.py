# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Grid World Environment.

This module creates an HTTP server that exposes the GridWorldEnvironment
via the OpenEnv API.
"""

# Import the correct app creation function from the core library
from core.env_server import create_fastapi_app

# Import our models and environment classes using relative paths
from ..models import GridWorldAction, GridWorldObservation
from .grid_world_environment import GridWorldEnvironment

# Create single environment instance
# This is reused for all HTTP requests.
env = GridWorldEnvironment()

# Create the FastAPI app
app = create_fastapi_app(
    env, 
    GridWorldAction, 
    GridWorldObservation
)