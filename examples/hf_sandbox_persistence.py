#!/usr/bin/env python3
"""Smoke-check persistent state through the HF sandbox provider.

This launches a tiny stateful WebSocket app on Hugging Face infrastructure,
sets a value over one WebSocket connection, reconnects, and reads it back over
a second connection.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from openenv.core.containers.runtime.hf_sandbox_provider import HFSandboxProvider
from websockets.asyncio.client import connect as ws_connect


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
    while True:
        message = await websocket.receive_json()
        if message["op"] == "set":
            state[message["name"]] = message["value"]
            await websocket.send_json({"ok": True})
        elif message["op"] == "get":
            await websocket.send_json({"value": state.get(message["name"])})
PY
cd /tmp && python -m uvicorn hf_sandbox_persistence_app:app --host 0.0.0.0 --port 8000
"""


def _ws_url(base_url: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :]
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :]
    return base_url


async def check_persistence(base_url: str, name: str, value: str) -> None:
    ws_url = f"{_ws_url(base_url)}/ws"

    async with ws_connect(ws_url) as websocket:
        await websocket.send(json.dumps({"op": "set", "name": name, "value": value}))
        set_response = json.loads(await websocket.recv())
        print(f"set response: {set_response}")

    async with ws_connect(ws_url) as websocket:
        await websocket.send(json.dumps({"op": "get", "name": name}))
        get_response = json.loads(await websocket.recv())
        print(f"get response: {get_response}")
        if get_response.get("value") != value:
            raise RuntimeError("HF sandbox did not preserve state across connections")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--flavor", default="cpu-basic")
    parser.add_argument("--namespace")
    parser.add_argument("--name", default="openenv_hf_sandbox_value")
    parser.add_argument("--value", default="persisted-across-connections")
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = HFSandboxProvider(
        flavor=args.flavor,
        namespace=args.namespace,
        command=SERVER_COMMAND,
        startup_timeout_s=args.startup_timeout_s,
    )

    with provider:
        base_url = provider.start_container(args.image)
        print(f"provider URL: {base_url}")
        provider.wait_for_ready(base_url, timeout_s=args.ready_timeout_s)
        asyncio.run(check_persistence(base_url, args.name, args.value))

    print("HF sandbox persistence check passed")


if __name__ == "__main__":
    main()
