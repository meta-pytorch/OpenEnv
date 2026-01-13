# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP server definition for Echo Environment.

Defines the MCP tools that echo_env exposes to agents.
"""

from fastmcp import FastMCP

# Create MCP server for echo_env
mcp = FastMCP("echo_env")


@mcp.tool
def echo_message(message: str) -> str:
    """
    Echo back the provided message.

    Args:
        message: The message to echo back

    Returns:
        The same message that was provided
    """
    return message


@mcp.tool
def echo_with_length(message: str) -> dict:
    """
    Echo back the message with its length.

    Args:
        message: The message to echo back

    Returns:
        Dictionary with the message and its length
    """
    return {
        "message": message,
        "length": len(message)
    }
