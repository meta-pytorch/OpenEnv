# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the WebSearch Env Environment.

This module creates an HTTP server that exposes the WebSearchEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv_core is required for the web interface. Install template deps with '\n"
        "    pip install -r server/requirements.txt\n'"
    ) from e

from .websearch_env_environment import WebSearchEnvironment
from ..models import WebSearchAction, WebSearchObservation

# Create the environment instance
env = WebSearchEnvironment()

# Create the app with web interface and README integration
app = create_app(env, WebSearchAction, WebSearchObservation, env_name="websearch_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
