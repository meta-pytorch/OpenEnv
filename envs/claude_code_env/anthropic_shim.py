# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Anthropic Messages -> OpenAI chat-completions translation shim.

Claude Code speaks the Anthropic Messages API only, but the interception proxy
(and vLLM) speak OpenAI chat-completions. This shim sits in front of the proxy:

    Claude Code --(Anthropic /v1/messages)--> shim --(OpenAI /v1/chat/completions)--> proxy --> vLLM

The shim is a pure translator. It never captures logprobs itself: it forwards a
non-streaming OpenAI request to the proxy, which injects ``logprobs`` and records
per-token ids + logprobs against vLLM exactly as it does for OpenAI-native agents
(opencode, pi). The shim then translates the full OpenAI reply back to the
Anthropic Messages shape and, when Claude Code asked to stream, replays it as a
synthetic Anthropic SSE sequence. Requesting the proxy unary keeps the capture
path identical to the other envs and avoids stream-reassembly on this side.

Run inside the sandbox:

    python anthropic_shim.py --upstream-url http://127.0.0.1:7000/v1 --port 7100
"""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# --- Anthropic <-> OpenAI stop-reason mapping --------------------------------
_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def _text_of(content: Any) -> str:
    """Flatten an Anthropic content value (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _translate_system(system: Any) -> list[dict]:
    """Anthropic ``system`` (str or list of text blocks) -> OpenAI system messages."""
    text = _text_of(system)
    return [{"role": "system", "content": text}] if text else []


def translate_request(body: dict) -> dict:
    """Translate an Anthropic Messages request body to an OpenAI chat-completions body."""
    messages: list[dict] = _translate_system(body.get("system"))

    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        # String content is the simple text case for either role.
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        blocks = content if isinstance(content, list) else []
        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in blocks:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
            out: dict[str, Any] = {"role": "assistant"}
            out["content"] = "".join(text_parts) or None
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
        else:  # user (or tool results carried on a user turn)
            text_parts = []
            for block in blocks:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    # Each Anthropic tool_result becomes its own OpenAI tool message.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": _text_of(block.get("content", "")),
                        }
                    )
            if text_parts:
                messages.append({"role": "user", "content": "".join(text_parts)})

    openai_body: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "stream": False,  # always unary to the proxy; the proxy captures on the full reply
    }
    if body.get("max_tokens") is not None:
        openai_body["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            openai_body[key] = body[key]
    if body.get("stop_sequences"):
        openai_body["stop"] = body["stop_sequences"]

    if body.get("tools"):
        openai_body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
            for tool in body["tools"]
        ]
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            ctype = choice.get("type")
            if ctype == "auto":
                openai_body["tool_choice"] = "auto"
            elif ctype == "any":
                openai_body["tool_choice"] = "required"
            elif ctype == "tool" and choice.get("name"):
                openai_body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice["name"]},
                }

    return openai_body


def translate_response(openai_resp: dict, model: str) -> dict:
    """Translate an OpenAI chat-completions response to an Anthropic Messages object."""
    choice = (openai_resp.get("choices") or [{}])[0]
    message = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")

    content_blocks: list[dict] = []
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name"),
                "input": args,
            }
        )

    usage = openai_resp.get("usage", {}) or {}
    return {
        "id": openai_resp.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": _FINISH_TO_STOP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_events(anthropic_msg: dict):
    """Replay a complete Anthropic message as a synthetic SSE event sequence."""
    msg_head = {k: anthropic_msg[k] for k in ("id", "type", "role", "model", "stop_sequence")}
    msg_head["content"] = []
    msg_head["stop_reason"] = None
    msg_head["usage"] = {"input_tokens": anthropic_msg["usage"]["input_tokens"], "output_tokens": 0}
    yield _sse("message_start", {"type": "message_start", "message": msg_head})

    for index, block in enumerate(anthropic_msg["content"]):
        if block["type"] == "text":
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            )
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}},
            )
        else:  # tool_use
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}},
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
                },
            )
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": anthropic_msg["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": anthropic_msg["usage"]["output_tokens"]},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def build_app(upstream_url: str, api_key: str) -> FastAPI:
    app = FastAPI()
    chat_url = upstream_url.rstrip("/") + "/chat/completions"
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(req: Request):
        body = await req.json()
        approx = len(json.dumps(body.get("messages", []))) // 4
        return {"input_tokens": max(1, approx)}

    @app.post("/v1/messages")
    async def messages(req: Request):
        body = await req.json()
        model = body.get("model", "unknown")
        want_stream = bool(body.get("stream"))
        openai_body = translate_request(body)

        try:
            resp = await client.post(
                chat_url, json=openai_body, headers={"Authorization": f"Bearer {api_key}"}
            )
            resp.raise_for_status()
            anthropic_msg = translate_response(resp.json(), model)
        except Exception as exc:  # surface upstream failures to Claude Code
            err = {"type": "error", "error": {"type": "api_error", "message": str(exc)}}
            if want_stream:
                return StreamingResponse(iter([_sse("error", err)]), media_type="text/event-stream")
            return JSONResponse(err, status_code=502)

        if want_stream:
            return StreamingResponse(stream_events(anthropic_msg), media_type="text/event-stream")
        return JSONResponse(anthropic_msg)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Anthropic Messages -> OpenAI translation shim")
    parser.add_argument("--upstream-url", required=True, help="OpenAI-compatible base URL (the interception proxy)")
    parser.add_argument("--api-key", default="intercepted", help="Bearer token forwarded to the upstream")
    parser.add_argument("--port", type=int, default=7100)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args.upstream_url, args.api_key), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
