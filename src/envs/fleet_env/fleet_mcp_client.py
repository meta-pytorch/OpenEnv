# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fleet-compatible MCP client wrapper (Streamable HTTP + initialize).

Design note:
- We intentionally avoid exposing an async context-manager interface here.
  Some MCP/AnyIO failure modes during connection setup can produce noisy
  ExceptionGroup/cancel-scope traces if a partially-entered context leaks.
- Instead, this wrapper provides *one-shot* operations that open + close the
  streamable HTTP transport within a single call.
"""

from typing import Any, Dict, List, Optional

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import Tool
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Fleet MCP support requires the optional dependency set. "
        "Install with `pip install openenv-core[fleet]`."
    ) from e


class FleetMCPClient:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

    async def list_tools(self) -> List[Tool]:
        async with streamablehttp_client(
            url=self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ) as streams:
            async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                await session.initialize()
                return (await session.list_tools()).tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        async with streamablehttp_client(
            url=self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ) as streams:
            async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return self._extract_tool_result(result)

    def _extract_tool_result(self, result: Any) -> Any:
        """Extract readable content from CallToolResult.

        MCP's call_tool returns a CallToolResult with content list.
        This extracts the text content for use in agent observations.
        """
        import json

        # Handle error case
        if hasattr(result, "isError") and result.isError:
            if hasattr(result, "content") and result.content:
                for content in result.content:
                    if hasattr(content, "text"):
                        return {"error": content.text}
            return {"error": "Tool execution failed"}

        # Extract content from CallToolResult
        if hasattr(result, "content") and result.content:
            texts = []
            for content in result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)
            if len(texts) == 1:
                # Single text result - try to parse as JSON
                try:
                    return json.loads(texts[0])
                except json.JSONDecodeError:
                    return texts[0]
            elif texts:
                # Multiple text results - return as list
                return texts

        # Fallback to structured content if available
        if hasattr(result, "structuredContent") and result.structuredContent:
            return result.structuredContent

        # Last resort - return string representation
        return str(result)

    def has_tool(self, name: str, tools_list: Optional[List[Tool]] = None) -> bool:
        if not tools_list:
            return False
        return any(t.name == name for t in tools_list)


