# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP server for Echo Environment.

This module defines the MCP tools exposed by the Echo environment.
Developers can add new tools by simply decorating functions with @mcp.tool.
"""

from fastmcp import FastMCP

mcp = FastMCP("echo_env")


@mcp.tool()
def echo_message(message: str) -> dict:
    """
    Echo a message back with metadata.

    Args:
        message: The message to echo

    Returns:
        Dictionary containing the echoed message and reward
    """
    return {
        "echoed_message": message,
        "reward": len(message) * 0.1,
    }
