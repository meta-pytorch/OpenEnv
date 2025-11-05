# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the __ENV_TITLE_NAME__ Environment.

This module creates an HTTP server that exposes the __ENV_CLASS_NAME__Environment
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
    from openenv_core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv_core is required for the web interface. Install template deps with '\n"
        "    pip install -r server/requirements.txt\n'"
    ) from e

from .__ENV_NAME___environment import __ENV_CLASS_NAME__Environment
from __ENV_NAME__.models import __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Observation

# Create the environment instance
env = __ENV_CLASS_NAME__Environment()

# Create the app with web interface and README integration
app = create_app(env, __ENV_CLASS_NAME__Action, __ENV_CLASS_NAME__Observation, env_name="__ENV_NAME__")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
