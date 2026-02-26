# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
AgentServer: HTTP server for a single agent process.

Runs inside a bwrap sandbox (or locally for development). Exposes:
  GET  /health       - Health check
  GET  /v1/info      - Process environment info (cwd, pid, mounts)
  POST /v1/turn      - Take a turn (SSE streaming response)
  POST /v1/history   - Retrieve conversation history
  POST /v1/control   - Runtime control operations (e.g. AddBundles)
  POST /v1/agentbus  - Query AgentBus log entries (if enabled)
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from aiohttp import web

from ..storage.uri import URIDownloader

logger = logging.getLogger(__name__)


def _load_bundles_from_dir(bundles_dir: Path) -> int:
    """Load Python bundles from a directory into sys.path.

    Scans the directory for subdirectories and adds each to sys.path
    so Python modules within them become importable.

    Returns the number of bundles loaded.
    """
    if not bundles_dir.exists():
        return 0

    loaded = 0
    for bundle_path in sorted(bundles_dir.iterdir()):
        if bundle_path.is_dir():
            sys.path.insert(0, str(bundle_path))
            logger.info("Loaded bundle: %s", bundle_path.name)
            loaded += 1
    return loaded


class AgentServer:
    """HTTP server exposing AgentService API for a single agent.

    Args:
        config: Agent configuration dict with at minimum "agent_id".
        llm_handler: Async callable that runs the LLM loop. Signature:
            (messages: list[dict], on_token: Callable) -> None
            This abstraction allows testing without real LLM calls.
    """

    def __init__(
        self,
        config: dict[str, Any],
        llm_handler: Callable[..., Any] | None = None,
        agentbus_helper: Any | None = None,
    ) -> None:
        self.agent_id: str = config["agent_id"]
        self.config = config
        self.history: list[dict[str, Any]] = []
        self._llm_handler = llm_handler
        self._agentbus_helper = agentbus_helper
        self._turn_lock = asyncio.Lock()

    async def start(self, port: int, host: str = "0.0.0.0") -> web.AppRunner:
        """Start the HTTP server. Returns the runner for cleanup."""
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/v1/info", self._handle_info)
        app.router.add_post("/v1/turn", self._handle_turn)
        app.router.add_post("/v1/history", self._handle_history)
        app.router.add_post("/v1/control", self._handle_control)
        if self._agentbus_helper is not None:
            app.router.add_post("/v1/agentbus", self._handle_agentbus)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info("AgentServer %s listening on %s:%d", self.agent_id, host, port)
        return runner

    # ── HTTP Handlers ─────────────────────────────────────────────────

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "agent_id": self.agent_id})

    async def _handle_info(self, request: web.Request) -> web.Response:
        """Report process environment — useful for verifying sandboxing."""
        info: dict[str, Any] = {
            "agent_id": self.agent_id,
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "uid": os.getuid(),
        }
        # /workspace as cwd is a strong signal of bwrap
        # PID 1 or low PIDs indicate PID namespace isolation
        try:
            info["root_contents"] = sorted(os.listdir("/"))
        except OSError:
            pass
        return web.json_response(info)

    async def _handle_turn(self, request: web.Request) -> web.StreamResponse:
        """Handle POST /v1/turn with SSE streaming response."""
        data = await request.json()

        if data.get("agent_id") != self.agent_id:
            logger.warning(
                "Turn rejected: wrong agent_id %s (expected %s)",
                data.get("agent_id"),
                self.agent_id,
            )
            return web.json_response(
                {"error": f"Wrong agent: expected {self.agent_id}"},
                status=400,
            )

        expected_nonce = self.config.get("nonce", "")
        if expected_nonce and data.get("nonce") != expected_nonce:
            logger.warning("Turn rejected: invalid nonce for agent %s", self.agent_id)
            return web.json_response(
                {"error": "Invalid nonce"},
                status=403,
            )

        if not self._llm_handler:
            logger.error("Turn rejected: no LLM handler configured")
            return web.json_response({"error": "No LLM handler configured"}, status=503)

        logger.info("Handling turn for agent %s", self.agent_id)

        # Prepare SSE response
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        # Stream tokens (serialize turns to prevent interleaved conversations)
        async with self._turn_lock:
            try:
                async for chunk in self._do_turn(data):
                    sse_line = f"data: {json.dumps(chunk)}\n\n"
                    await response.write(sse_line.encode())
            except Exception as e:
                error_chunk = {
                    "body": "",
                    "done": True,
                    "error": str(e),
                }
                await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())

        return response

    async def _handle_history(self, request: web.Request) -> web.Response:
        data = await request.json()
        last_n = data.get("last_n", 0)
        entries = self.history[-last_n:] if last_n > 0 else self.history
        return web.json_response({"entries": entries})

    async def _handle_control(self, request: web.Request) -> web.Response:
        """Handle POST /v1/control for runtime control operations."""
        data = await request.json()

        # Validate nonce
        expected_nonce = self.config.get("nonce", "")
        if expected_nonce and data.get("nonce") != expected_nonce:
            logger.warning(
                "Control rejected: invalid nonce for agent %s", self.agent_id
            )
            return web.json_response({"error": "Invalid nonce"}, status=403)

        op = data.get("op")
        if op == "add_bundles":
            return await self._handle_add_bundles(data)
        else:
            return web.json_response(
                {"error": f"Unknown control operation: {op}"}, status=400
            )

    async def _handle_add_bundles(self, data: dict[str, Any]) -> web.Response:
        """Handle the add_bundles control operation.

        Downloads bundles from URIs and adds them to sys.path for import.
        """
        bundles = data.get("bundles", [])
        if not bundles:
            return web.json_response({"error": "No bundles specified"}, status=400)

        downloader = URIDownloader()
        workspace_bundles = Path("/workspace/bundles")
        workspace_bundles.mkdir(parents=True, exist_ok=True)

        loaded = []
        for bundle in bundles:
            uri = bundle.get("uri", "")
            labels = bundle.get("labels", {})
            name = labels.get("name", "bundle")
            dest = workspace_bundles / name

            try:
                await downloader.download(uri, dest)
                sys.path.insert(0, str(dest))
                loaded.append(name)
                logger.info("Hot-loaded bundle: %s from %s", name, uri)
            except Exception as e:
                logger.error("Failed to load bundle %s: %s", name, e)
                return web.json_response(
                    {"error": f"Failed to load bundle {name}: {e}"}, status=500
                )

        return web.json_response({"status": "ok", "loaded": loaded})

    # ── AgentBus query ────────────────────────────────────────────────

    async def _handle_agentbus(self, request: web.Request) -> web.Response:
        """Handle POST /v1/agentbus — query AgentBus log entries."""
        data = await request.json()

        # Validate nonce
        expected_nonce = self.config.get("nonce", "")
        if expected_nonce and data.get("nonce") != expected_nonce:
            logger.warning(
                "AgentBus query rejected: invalid nonce for agent %s",
                self.agent_id,
            )
            return web.json_response({"error": "Invalid nonce"}, status=403)

        if not self._agentbus_helper:
            return web.json_response(
                {"error": "AgentBus not configured for this agent"}, status=503
            )

        from agentbus.proto.agent_bus_pb2 import (  # pyre-ignore[21]
            PayloadTypeFilter,
            PollRequest,
            SelectivePollType,
        )

        client = self._agentbus_helper.client
        bus_id = client._config.bus_id

        start_position = int(data.get("start_position", 0))
        payload_types = data.get("payload_types", [])

        # Map string type names → SelectivePollType enum values
        # pyre-ignore[16]: SelectivePollType comes from generated proto stubs
        type_name_to_enum = {
            "intention": SelectivePollType.INTENTION,  # pyre-ignore[16]
            "vote": SelectivePollType.VOTE,  # pyre-ignore[16]
            "decider_policy": SelectivePollType.DECIDER_POLICY,  # pyre-ignore[16]
            "commit": SelectivePollType.COMMIT,  # pyre-ignore[16]
            "abort": SelectivePollType.ABORT,  # pyre-ignore[16]
            "voter_policy": SelectivePollType.VOTER_POLICY,  # pyre-ignore[16]
            "control": SelectivePollType.CONTROL,  # pyre-ignore[16]
            "inference_input": SelectivePollType.INFERENCE_INPUT,  # pyre-ignore[16]
            "inference_output": SelectivePollType.INFERENCE_OUTPUT,  # pyre-ignore[16]
            "action_output": SelectivePollType.ACTION_OUTPUT,  # pyre-ignore[16]
            "agent_input": SelectivePollType.AGENT_INPUT,  # pyre-ignore[16]
            "agent_output": SelectivePollType.AGENT_OUTPUT,  # pyre-ignore[16]
        }

        # Build optional filter
        poll_filter = None
        if payload_types:
            enum_types = [
                type_name_to_enum[t] for t in payload_types if t in type_name_to_enum
            ]
            if enum_types:
                poll_filter = PayloadTypeFilter(  # pyre-ignore[16]
                    payload_types=enum_types
                )

        # Poll entries (server returns max 64 per page)
        entries: list[dict[str, Any]] = []
        pos = start_position
        max_pages = 100
        for _ in range(max_pages):
            kwargs: dict[str, Any] = {
                "agent_bus_id": bus_id,
                "start_log_position": pos,
                "max_entries": 64,
            }
            if poll_filter is not None:
                kwargs["filter"] = poll_filter
            poll_resp = await client._stub.Poll(
                PollRequest(**kwargs)  # pyre-ignore[16]
            )

            for entry in poll_resp.entries:
                payload = entry.payload
                which = payload.WhichOneof("payload")
                entries.append(
                    {
                        "position": entry.header.log_position,
                        "type": which or "",
                        "payload": self._extract_payload_string(payload, which),
                    }
                )
                pos = entry.header.log_position + 1

            if poll_resp.complete or not poll_resp.entries:
                break

        return web.json_response({"entries": entries})

    @staticmethod
    def _extract_payload_string(payload: Any, which: str | None) -> str:
        """Extract a readable string from a Payload's active oneof field."""
        if which is None:
            return ""
        sub = getattr(payload, which)

        # Most types have a single string oneof field.
        _string_fields = {
            "intention": "string_intention",
            "inference_input": "string_inference_input",
            "inference_output": "string_inference_output",
            "action_output": "string_action_output",
            "agent_input": "string_agent_input",
            "agent_output": "string_agent_output",
            "voter_policy": "prompt_override",
        }
        if which in _string_fields:
            return getattr(sub, _string_fields[which], "")

        # Structured types — serialise key fields.
        if which in ("commit", "abort"):
            return json.dumps({"intention_id": sub.intention_id, "reason": sub.reason})
        if which == "control":
            ctrl_which = sub.WhichOneof("control")
            if ctrl_which == "agent_input":
                return sub.agent_input
            return ""

        # Fallback (vote, decider_policy, …)
        return str(sub)

    # ── Core Logic ────────────────────────────────────────────────────

    async def _do_turn(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Execute a turn: bridge LLM handler streaming to SSE chunks.

        Accumulates conversation history so each turn sees the full
        conversation — enabling multi-turn follow-on conversations.
        """
        body = request.get("body", "")

        # Parse new messages from body
        if isinstance(body, str):
            try:
                body_parsed = json.loads(body)
            except json.JSONDecodeError:
                body_parsed = {"messages": [{"role": "user", "content": body}]}
        else:
            body_parsed = body

        new_messages = body_parsed.get("messages", [])

        # Record incoming messages
        for msg in new_messages:
            self.history.append(
                {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        # Build full conversation for the LLM handler
        conversation = [
            {"role": entry["role"], "content": entry["content"]}
            for entry in self.history
        ]

        # Queue for streaming tokens
        token_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        accumulated: list[str] = []
        error_holder: list[Exception] = []

        async def on_token(token: Any) -> None:
            text = (
                token if isinstance(token, str) else getattr(token, "text", str(token))
            )
            accumulated.append(text)
            await token_queue.put({"body": text, "done": False})

        async def run_llm() -> None:
            assert self._llm_handler is not None
            try:
                await self._llm_handler(conversation, on_token)
            except Exception as e:
                logger.error(
                    "LLM handler failed for agent %s: %s",
                    self.agent_id,
                    e,
                    exc_info=True,
                )
                error_holder.append(e)
            finally:
                done_chunk: dict[str, Any] = {"body": "", "done": True}
                if error_holder:
                    done_chunk["error"] = str(error_holder[0])
                await token_queue.put(done_chunk)

        task = asyncio.create_task(run_llm())

        try:
            while True:
                chunk = await token_queue.get()
                yield chunk
                if chunk["done"]:
                    break
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Record assistant response
        self.history.append(
            {
                "role": "assistant",
                "content": "".join(accumulated),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
