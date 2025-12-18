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
from .models import ListToolsAction, convert_tool_format


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

    async def list_tools(self) -> ListToolsAction:
        """List available tools (union across endpoints) as a ListToolsAction.

        The returned `.tools` payload is in OpenAI "tools" dict format
        (see `convert_tool_format`), derived from MCP `Tool.inputSchema`.
        """
        owner_cache = self._get_owner_cache()
        tools: list[Any] = []
        seen: set[str] = set()
        for client in self._get_clients():
            try:
                found = await client.list_tools()
                for t in found:
                    # Deduplicate by tool name across endpoints, but cache first-seen owner.
                    if t.name not in owner_cache:
                        owner_cache[t.name] = client
                    if t.name in seen:
                        continue
                    seen.add(t.name)
                    tools.append(convert_tool_format(t))
            except BaseException:  # noqa: BLE001
                continue
        return ListToolsAction(tools=tools)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        owner_cache = self._get_owner_cache()
        clients = self._get_clients()

        if tool_name in owner_cache:
            client = owner_cache[tool_name]
            return await client.call_tool(tool_name, arguments)

        for client in clients:
            try:
                tools = await client.list_tools()
                if client.has_tool(tool_name, tools):
                    owner_cache[tool_name] = client
                    # If execution fails here, we let it propagate because we found the owner.
                    return await client.call_tool(tool_name, arguments)
            except BaseException:
                # Only suppress discovery/connection errors.
                # If call_tool raised, it would have bubbled up above.
                continue

        raise ValueError(f"Tool '{tool_name}' not found on any active MCP endpoint.")


