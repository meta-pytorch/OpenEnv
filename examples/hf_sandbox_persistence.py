#!/usr/bin/env python3
"""Smoke-check persistent state through the HF sandbox provider.

This launches a tiny stateful WebSocket app on Hugging Face infrastructure,
sets a value over one WebSocket connection, reconnects, and reads it back over
a second connection.
"""

from __future__ import annotations

import asyncio
import json

from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider
from websockets.asyncio.client import connect as ws_connect


IMAGE = "python:3.12-slim"
STATE_NAME = "openenv_hf_sandbox_value"
STATE_VALUE = "persisted-across-connections"
SERVER_COMMAND = r"""
python -m pip install --quiet --disable-pip-version-check fastapi "uvicorn[standard]"
cat > /tmp/hf_sandbox_persistence_app.py <<'PY'
from fastapi import FastAPI, WebSocket

app = FastAPI()
state = {}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async for message in websocket.iter_json():
        if message["op"] == "set":
            state[message["name"]] = message["value"]
            await websocket.send_json({"ok": True})
        elif message["op"] == "get":
            await websocket.send_json({"value": state.get(message["name"])})
PY
cd /tmp && python -m uvicorn hf_sandbox_persistence_app:app --host 0.0.0.0 --port 8000
"""


async def check_persistence(base_url: str) -> None:
    ws_url = f"{base_url.replace('http://', 'ws://', 1)}/ws"

    async with ws_connect(ws_url) as websocket:
        await websocket.send(
            json.dumps({"op": "set", "name": STATE_NAME, "value": STATE_VALUE})
        )
        set_response = json.loads(await websocket.recv())
        print(f"set response: {set_response}")

    async with ws_connect(ws_url) as websocket:
        await websocket.send(json.dumps({"op": "get", "name": STATE_NAME}))
        get_response = json.loads(await websocket.recv())
        print(f"get response: {get_response}")
        if get_response.get("value") != STATE_VALUE:
            raise RuntimeError("HF sandbox did not preserve state across connections")


def main() -> None:
    provider = HFSandboxProvider(command=SERVER_COMMAND)

    with provider:
        base_url = provider.start_container(IMAGE)
        print(f"provider URL: {base_url}")
        provider.wait_for_ready(base_url, timeout_s=300.0)
        asyncio.run(check_persistence(base_url))

    print("HF sandbox persistence check passed")


if __name__ == "__main__":
    main()
