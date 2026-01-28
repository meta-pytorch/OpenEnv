# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MCP-only handle for agents (no reset/step/state)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .fleet_mcp_client import FleetMCPClient
from .models import ListToolsAction, convert_tool_format

logger = logging.getLogger(__name__)


@dataclass
class FleetMCPTools:
    """Agent-facing tools client (MCP only)."""

    api_key: str
    mcp_urls: Sequence[str]
    max_retries: int = 3
    retry_base_delay: float = 1.0
    _clients: Optional[List[FleetMCPClient]] = field(default=None, repr=False)
    _tool_owner: Optional[Dict[str, FleetMCPClient]] = field(default=None, repr=False)

    def _get_clients(self) -> List[FleetMCPClient]:
        if self._clients is None:
            self._clients = [FleetMCPClient(url, self.api_key) for url in self.mcp_urls]
        return self._clients

    def _get_owner_cache(self) -> Dict[str, FleetMCPClient]:
        if self._tool_owner is None:
            self._tool_owner = {}
        return self._tool_owner

    async def _list_tools_single_attempt(self) -> List[Any]:
        """Single attempt to list tools from all clients."""
        owner_cache = self._get_owner_cache()
        tools: list[Any] = []
        seen: set[str] = set()
        errors: list[str] = []

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
            except BaseException as e:
                errors.append(f"{client.url}: {e}")
                continue

        if errors and not tools:
            # All clients failed - log and raise
            raise RuntimeError(f"All MCP clients failed to list tools: {errors}")

        if errors:
            # Some clients failed but we got tools from others
            logger.warning(f"Some MCP clients failed to list tools: {errors}")

        return tools

    async def list_tools(self) -> ListToolsAction:
        """List available tools (union across endpoints) as a ListToolsAction.

        The returned `.tools` payload is in OpenAI "tools" dict format
        (see `convert_tool_format`), derived from MCP `Tool.inputSchema`.

        Retries with exponential backoff if all clients fail.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                tools = await self._list_tools_single_attempt()
                if tools:
                    return ListToolsAction(tools=tools)
                # Got empty tools - treat as failure and retry
                raise RuntimeError("No tools found from any MCP endpoint")
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"list_tools attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        logger.error(f"list_tools failed after {self.max_retries} attempts: {last_error}")
        return ListToolsAction(tools=[])

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        owner_cache = self._get_owner_cache()
        clients = self._get_clients()

        if tool_name in owner_cache:
            client = owner_cache[tool_name]
            return await client.call_tool(tool_name, arguments)

        errors: list[str] = []
        for client in clients:
            try:
                tools = await client.list_tools()
                if client.has_tool(tool_name, tools):
                    owner_cache[tool_name] = client
                    # If execution fails here, we let it propagate because we found the owner.
                    return await client.call_tool(tool_name, arguments)
            except BaseException as e:
                # Log discovery/connection errors instead of silently swallowing.
                errors.append(f"{client.url}: {e}")
                continue

        if errors:
            logger.warning(f"Some MCP clients failed during tool discovery: {errors}")

        raise ValueError(f"Tool '{tool_name}' not found on any active MCP endpoint.")


