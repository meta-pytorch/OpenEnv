# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""AgentClient - shared HTTP+SSE transport for talking to running agents."""

import json
import logging
from typing import AsyncIterator

import aiohttp

from ..config import TurnRequest, TurnResponse
from .protocol import Resolver

logger = logging.getLogger(__name__)


class AgentClient:
    """Client for talking to running agents via HTTP+SSE.

    Backend-agnostic — routing is handled by the injected Resolver.
    """

    def __init__(self, resolver: Resolver) -> None:
        self._resolver = resolver

    async def turn(self, request: TurnRequest) -> AsyncIterator[TurnResponse]:
        """Send a turn request to an agent and stream back responses."""
        base_url = await self._resolver.resolve(request.agent_id)
        url = f"{base_url}/v1/turn"
        logger.info("Sending turn to %s", url)

        payload = {
            "agent_id": request.agent_id,
            "nonce": request.nonce,
            "body": request.body.decode()
            if isinstance(request.body, bytes)
            else request.body,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Turn request failed: %d %s", resp.status, body)
                    yield TurnResponse(
                        body="",
                        done=True,
                        error=f"HTTP {resp.status}: {body}",
                    )
                    return
                async for line in resp.content:
                    decoded = line.decode().strip()
                    if not decoded.startswith("data: "):
                        continue
                    try:
                        data = json.loads(decoded[6:])
                        error = data.get("error")
                        if error:
                            logger.error(
                                "Agent %s returned error: %s", request.agent_id, error
                            )
                        yield TurnResponse(
                            body=data.get("body", ""),
                            done=data.get("done", False),
                            error=error,
                        )
                        if data.get("done"):
                            return
                    except json.JSONDecodeError:
                        continue

    async def get_info(self, agent_id: str) -> dict:
        """Get process environment info from an agent."""
        base_url = await self._resolver.resolve(agent_id)
        url = f"{base_url}/v1/info"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.json()

    async def get_history(self, agent_id: str, last_n: int = 0) -> list[dict]:
        """Get conversation history from an agent."""
        base_url = await self._resolver.resolve(agent_id)
        url = f"{base_url}/v1/history"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"last_n": last_n}) as resp:
                data = await resp.json()
                return data.get("entries", [])
