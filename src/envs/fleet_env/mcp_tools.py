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


def _unwrap_exception(e: Exception) -> str:
    """Extract meaningful error message from ExceptionGroup or nested exceptions."""
    # Handle ExceptionGroup (from asyncio.TaskGroup)
    if hasattr(e, 'exceptions'):
        msgs = [_unwrap_exception(sub) for sub in e.exceptions]
        return "; ".join(msgs)
    # Handle chained exceptions
    if e.__cause__:
        return f"{type(e).__name__}: {e} <- {_unwrap_exception(e.__cause__)}"
    return f"{type(e).__name__}: {e}"


@dataclass
class FleetMCPTools:
    """Agent-facing tools client (MCP only)."""

    api_key: str
    mcp_urls: Sequence[str]
    max_retries: int = 3
    retry_base_delay: float = 2.0
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
                error_msg = _unwrap_exception(e)
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"list_tools attempt {attempt + 1}/{self.max_retries} failed: {error_msg}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        logger.error(f"list_tools failed after {self.max_retries} attempts: {_unwrap_exception(last_error)}")
        raise RuntimeError(
            f"list_tools failed after {self.max_retries} attempts"
        ) from last_error

    async def _call_tool_single_attempt(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Single attempt to call a tool."""
        owner_cache = self._get_owner_cache()
        clients = self._get_clients()

        if tool_name in owner_cache:
            client = owner_cache[tool_name]
            logger.debug(f"call_tool({tool_name}) using cached client: {client.url}")
            return await client.call_tool(tool_name, arguments)

        errors: list[str] = []
        for client in clients:
            try:
                tools = await client.list_tools()
                if client.has_tool(tool_name, tools):
                    owner_cache[tool_name] = client
                    return await client.call_tool(tool_name, arguments)
            except BaseException as e:
                errors.append(f"{client.url}: {e}")
                continue

        if errors:
            raise RuntimeError(f"Tool call failed: {errors}")

        raise ValueError(f"Tool '{tool_name}' not found on any active MCP endpoint.")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool with retry logic for connection failures.

        Retries with exponential backoff on connection errors.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await self._call_tool_single_attempt(tool_name, arguments)
                if attempt > 0:
                    logger.info(f"call_tool({tool_name}) succeeded on attempt {attempt + 1}")
                return result
            except ValueError:
                # Tool not found - don't retry
                raise
            except Exception as e:
                last_error = e
                error_msg = _unwrap_exception(e)
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay * (2**attempt)
                    logger.warning(
                        f"call_tool({tool_name}) attempt {attempt + 1}/{self.max_retries} failed: {error_msg}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

        logger.error(
            f"call_tool({tool_name}) failed after {self.max_retries} attempts: {_unwrap_exception(last_error)}"
        )
        raise RuntimeError(
            f"call_tool({tool_name}) failed after {self.max_retries} attempts"
        ) from last_error


