# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Coding Environment.

This module creates an HTTP server that exposes the PythonCodeActEnv
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.coding_env.server.app:app --reload --host 0.0.0.0 --port 5432

    # Production:
    uvicorn envs.coding_env.server.app:app --host 0.0.0.0 --port 5432 --workers 4

    # Or run directly:
    python -m envs.coding_env.server.app
    
    # With custom imports:
    PYTHON_ADDITIONAL_IMPORTS="sys,os,functools,typing" python -m envs.coding_env.server.app
"""

import os

from core.env_server import create_app

from ..models import CodeAction, CodeObservation
from .python_codeact_env import PythonCodeActEnv

# Get additional imports from environment variable
# Format: comma-separated list, e.g., "sys,os,functools,typing"
additional_imports_str = os.environ.get("PYTHON_ADDITIONAL_IMPORTS", "")
if additional_imports_str:
    additional_imports = [imp.strip() for imp in additional_imports_str.split(",") if imp.strip()]
else:
    # Default imports that match the common_imports used in reward evaluation
    additional_imports = [
        "sys",
        "os",
        "functools",
        "typing",
    ]

# Create the environment instance with authorized imports
env = PythonCodeActEnv(additional_imports=additional_imports)

# Create the app with web interface and README integration
app = create_app(env, CodeAction, CodeObservation, env_name="coding_env")


if __name__ == "__main__":
    import sys
    import uvicorn

    # Get port from environment variable or command line argument
    # Priority: command line arg > environment variable > default (5432)
    port = int(os.environ.get("PORT", 5432))
    
    # Override with command line argument if provided
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port argument: {sys.argv[1]}, using port {port}")
    
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
