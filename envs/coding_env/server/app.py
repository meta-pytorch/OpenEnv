# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Coding Environment with MCP Support.

This module creates an HTTP server that exposes the PythonCodeActEnv
over HTTP, WebSocket, and MCP endpoints.

The environment supports two interaction styles:
1. **MCP Tools** (recommended): Use ListToolsAction/CallToolAction or POST /mcp
   - execute_code(code): Execute Python code
   - check_syntax(code): Check syntax without executing

2. **Legacy CodeAction**: For backward compatibility with existing clients

Usage:
    # Development (with auto-reload):
    uvicorn envs.coding_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.coding_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Using MCP client:
    from openenv.core.mcp_client import MCPToolClient
    with MCPToolClient(base_url="http://localhost:8000") as env:
        env.reset()
        result = env.call_tool("execute_code", code="print('Hello!')")
"""

from openenv.core.env_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from coding_env.server.python_codeact_env import PythonCodeActEnv

# Create the app with MCP support
# Using CallToolAction/CallToolObservation for MCP compatibility
# Legacy CodeAction is still supported via _step_impl
app = create_app(
    PythonCodeActEnv,
    CallToolAction,
    CallToolObservation,
    env_name="coding_env",
)


def main():
    """Main entry point for running the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
