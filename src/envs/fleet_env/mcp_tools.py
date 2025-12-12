# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MCP-only handle for agents (no reset/step/state)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .fleet_mcp_client import FleetMCPClient


@dataclass
class FleetMCPTools:
    """Agent-facing tools client (MCP only)."""

    api_key: str
    mcp_urls: Sequence[str]
    _clients: Optional[List[FleetMCPClient]] = None
    _tool_owner: Optional[Dict[str, FleetMCPClient]] = None

    def _get_clients(self) -> List[FleetMCPClient]:
        if self._clients is None:
            self._clients = [FleetMCPClient(url, self.api_key) for url in self.mcp_urls]
        return self._clients

    def _get_owner_cache(self) -> Dict[str, FleetMCPClient]:
        if self._tool_owner is None:
            self._tool_owner = {}
        return self._tool_owner

    async def list_tools(self) -> list[Any]:
        tools: list[Any] = []
        for client in self._get_clients():
            try:
                async with client:
                    tools.extend(await client.list_tools())
            except Exception:  # noqa: BLE001
                continue
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        owner_cache = self._get_owner_cache()
        clients = self._get_clients()

        if tool_name in owner_cache:
            client = owner_cache[tool_name]
            async with client:
                return await client.call_tool(tool_name, arguments)

        for client in clients:
            try:
                async with client:
                    tools = await client.list_tools()
                    if client.has_tool(tool_name, tools):
                        owner_cache[tool_name] = client
                        # If execution fails here, we let it propagate because we found the owner.
                        return await client.call_tool(tool_name, arguments)
            except Exception:
                # Only suppress discovery/connection errors.
                # If call_tool raised, it would have bubbled up above.
                continue

        raise ValueError(f"Tool '{tool_name}' not found on any active MCP endpoint.")


