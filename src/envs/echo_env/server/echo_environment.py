# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment Implementation (MCP-based).

A simple test environment that echoes back messages via MCP tools.
Perfect for testing HTTP server and MCP infrastructure.
"""

try:
    from core.env_server import MCPEnvironment
except ImportError:
    from openenv_core.env_server import MCPEnvironment

from .mcp_server import mcp


class EchoEnvironment(MCPEnvironment):
    """
    A simple echo environment that echoes back messages via MCP tools.

    This environment demonstrates the simplified MCP integration pattern.
    All functionality is defined in mcp_server.py using FastMCP decorators,
    and MCPEnvironment handles the rest.

    Example:
        >>> from envs.echo_env.server import EchoEnvironment
        >>> from core.env_server import create_fastapi_app
        >>> from core.env_server.mcp_types import Action, Observation
        >>>
        >>> env = EchoEnvironment()
        >>> app = create_fastapi_app(env, Action, Observation)
        >>>
        >>> # Run with: uvicorn app:app --port 8000
    """

    def __init__(self):
        """Initialize the echo environment with the MCP server."""
        super().__init__(mcp_server=mcp)
