# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fleet-compatible MCP client wrapper (Streamable HTTP + initialize)."""

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
        self._exit_stack = None
        self._session: Optional[ClientSession] = None

    async def __aenter__(self):
        from contextlib import AsyncExitStack

        self._exit_stack = AsyncExitStack()
        streams = await self._exit_stack.enter_async_context(
            streamablehttp_client(
                url=self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        )

        if len(streams) == 2:
            read_stream, write_stream = streams
        else:
            read_stream, write_stream = streams[0], streams[1]

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream=read_stream, write_stream=write_stream)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack:
            await self._exit_stack.aclose()
        self._session = None

    async def list_tools(self) -> List[Tool]:
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with'.")
        return (await self._session.list_tools()).tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if not self._session:
            raise RuntimeError("Client not connected. Use 'async with'.")
        return await self._session.call_tool(name, arguments)

    def has_tool(self, name: str, tools_list: Optional[List[Tool]] = None) -> bool:
        if not tools_list:
            return False
        return any(t.name == name for t in tools_list)


