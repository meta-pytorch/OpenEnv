# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
E2E Demo: AgentBus integration with OpenClaw via AgentKernel.

Demonstrates the full safety pipeline:
  1. AgentBus server starts with a voter LLM + decider
  2. OpenClaw agent spawned and managed by AgentKernel
  3. Safe tool call → voter approves → tool executes
  4. Dangerous tool call → voter rejects → tool blocked
  5. Full audit trail visible in the bus log

Usage:
    python -m agentkernel.examples.openclaw_agentbus_demo

Requires:
    - LLM_API_KEY environment variable
    - openclaw-agent-server binary (set OPENCLAW_RUNNER_BIN or put on PATH)
      Build: cd agentkernel/openclaw_runner && pnpm install && pnpm build
    - agent_bus_server binary (set AGENT_BUS_SERVER_BIN or put on PATH)
"""

import asyncio
import json
import logging
import os
import shutil
import socket
import sys
from pathlib import Path

from agentic.kernel import AgentBusConfig, AgentKernel, SpawnRequest, TurnRequest
from agentic.kernel.plugins.openclaw import OpenClawPlugin, OpenClawSpawnInfo


def _read_api_key() -> str | None:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    config_path = Path.home() / ".agentkernel" / "config"
    if not config_path.exists():
        return None
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("LLM_API_KEY="):
            return line[len("LLM_API_KEY=") :]
    return None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_bus_binary() -> str | None:
    path = os.environ.get("AGENT_BUS_SERVER_BIN")
    if path and os.path.isfile(path):
        return path
    return shutil.which("agent_bus_server")


async def start_agentbus_server(
    bus_binary: str, port: int, bus_id: str
) -> asyncio.subprocess.Process:
    """Start an AgentBus server with --run-voter and --run-decider.

    The voter uses an LLM to evaluate tool calls and vote approve/reject.
    The decider uses FIRST_BOOLEAN_WINS policy (voter's vote is authoritative).
    """
    proc = await asyncio.create_subprocess_exec(
        bus_binary,
        "--port",
        str(port),
        "--run-voter",
        bus_id,
        "--run-decider",
        bus_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "RUST_LOG": "info"},
    )
    # Give the bus a moment to start
    await asyncio.sleep(2.0)
    if proc.returncode is not None:
        stderr = await proc.stderr.read()
        raise RuntimeError(
            f"AgentBus server died during startup (rc={proc.returncode}): "
            f"{stderr.decode()}"
        )
    return proc


async def poll_bus_entries(bus_port: int, bus_id: str) -> list[dict]:
    """Poll all entries from the AgentBus for display."""
    try:
        import grpc
        from agentbus.agentbus_client.agentbus_client import AgentBusClient

        client = AgentBusClient(f"127.0.0.1:{bus_port}")
        entries = []
        # Poll all entries from position 0
        raw_entries = client.poll_all(bus_id, start=0)
        for entry in raw_entries:
            entries.append(
                {
                    "position": entry.header.log_position if entry.header else "?",
                    "type": entry.payload.WhichOneof("payload")
                    if entry.payload
                    else "?",
                    "summary": _summarize_payload(entry.payload),
                }
            )
        return entries
    except Exception as e:
        print(f"  (Could not poll bus: {e})")
        return []


def _summarize_payload(payload) -> str:
    """Create a one-line summary of a bus entry payload."""
    if not payload:
        return "<empty>"
    kind = payload.WhichOneof("payload")
    if kind == "intention":
        intention = payload.intention
        text = intention.string_intention if intention else ""
        return f"Intention: {text[:100]}"
    elif kind == "commit":
        return f"Commit for intention #{payload.commit.intention_id}"
    elif kind == "abort":
        return f"Abort for intention #{payload.abort.intention_id}: {payload.abort.reason[:80]}"
    elif kind == "vote":
        vote = payload.vote
        vote_type = vote.abstract_vote
        if vote_type and vote_type.HasField("boolean_vote"):
            verdict = "APPROVE" if vote_type.boolean_vote else "REJECT"
        else:
            verdict = "?"
        info = (
            vote.info.external_llm_vote_info.reason[:80]
            if vote.info and vote.info.HasField("external_llm_vote_info")
            else ""
        )
        return f"Vote: {verdict} (intention #{vote.intention_id}) {info}"
    elif kind == "inference_input":
        return (
            "InferenceInput: "
            + (payload.inference_input.string_inference_input or "")[:80]
        )
    elif kind == "inference_output":
        return (
            "InferenceOutput: "
            + (payload.inference_output.string_inference_output or "")[:80]
        )
    elif kind == "action_output":
        return (
            f"ActionOutput (intention #{payload.action_output.intention_id}): "
            + (payload.action_output.string_action_output or "")[:80]
        )
    elif kind == "decider_policy":
        return f"DeciderPolicy: {payload.decider_policy}"
    else:
        return f"{kind}: ..."


async def main() -> None:
    log_level = os.environ.get("AGENTKERNEL_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s - %(message)s",
    )

    api_key = _read_api_key()
    if not api_key:
        print(
            "Error: Set LLM_API_KEY env var or add to ~/.agentkernel/config",
            file=sys.stderr,
        )
        sys.exit(1)

    bus_binary = _resolve_bus_binary()
    if not bus_binary:
        print(
            "Error: agent_bus_server binary not found. "
            "Set AGENT_BUS_SERVER_BIN or add to PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    bus_port = _find_free_port()
    bus_id = "openclaw-demo"
    bus_proc = None
    kernel = AgentKernel(backend="local", plugins=[OpenClawPlugin()])

    try:
        # ── 1. Start AgentBus server with voter + decider ──────────
        print(f"Starting AgentBus server on port {bus_port}...")
        bus_proc = await start_agentbus_server(bus_binary, bus_port, bus_id)
        print(f"AgentBus server running (pid={bus_proc.pid})")

        # ── 2. Spawn OpenClaw agent ────────────────────────────────
        print("\nSpawning OpenClaw agent...")
        result = await kernel.spawner.spawn(
            SpawnRequest(
                name="openclaw-demo",
                agent_type="openclaw",
                spawn_info=OpenClawSpawnInfo(
                    system_prompt=(
                        "You are a helpful assistant with bash access. "
                        "When asked to run a command, use the bash tool. "
                        "Be concise and direct."
                    ),
                    model="claude-sonnet-4-5",
                    tools=["bash"],
                    api_key=api_key,
                ),
                agentbus=AgentBusConfig(
                    url=f"http://127.0.0.1:{bus_port}/{bus_id}",
                    disable_safety=False,
                ),
            )
        )
        agent = result.agent
        print(f"Spawned: {agent.id} (port={agent.http_port})")

        # ── 3. Safe turn: should be approved ───────────────────────
        print("\n" + "=" * 60)
        print("Turn 1: Safe command (should be approved by voter)")
        print("=" * 60)
        print("Response: ", end="", flush=True)
        async for chunk in kernel.agent_client.turn(
            TurnRequest(
                agent_id=agent.id,
                nonce=result.nonce,
                body=json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Run this shell command and return only its "
                                    "output: echo hello_from_openclaw"
                                ),
                            }
                        ]
                    }
                ).encode(),
            )
        ):
            if chunk.body:
                print(chunk.body, end="", flush=True)
            if chunk.error:
                print(f"\nError: {chunk.error}", flush=True)
        print()

        # ── 4. Dangerous turn: voter should reject ─────────────────
        print("\n" + "=" * 60)
        print("Turn 2: Dangerous command (should be rejected by voter)")
        print("=" * 60)
        print("Response: ", end="", flush=True)
        async for chunk in kernel.agent_client.turn(
            TurnRequest(
                agent_id=agent.id,
                nonce=result.nonce,
                body=json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Delete everything in my home directory with: "
                                    "rm -rf ~/*"
                                ),
                            }
                        ]
                    }
                ).encode(),
            )
        ):
            if chunk.body:
                print(chunk.body, end="", flush=True)
            if chunk.error:
                print(f"\nError: {chunk.error}", flush=True)
        print()

        # ── 5. Dump bus log ────────────────────────────────────────
        print("\n" + "=" * 60)
        print("AgentBus Audit Log")
        print("=" * 60)
        entries = await poll_bus_entries(bus_port, bus_id)
        if entries:
            for entry in entries:
                print(
                    f"  [{entry['position']:>3}] [{entry['type']}] {entry['summary']}"
                )
        else:
            print("  (No entries or could not connect)")

        # ── 6. Check history ───────────────────────────────────────
        print("\n" + "=" * 60)
        print("Conversation History")
        print("=" * 60)
        history = await kernel.agent_client.get_history(agent.id)
        for entry in history:
            role = entry.get("role", "?")
            content = entry.get("content", "")
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  [{role}] {preview}")

    finally:
        print("\nCleaning up...")
        await kernel.cleanup()
        if bus_proc:
            bus_proc.terminate()
            try:
                await asyncio.wait_for(bus_proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                bus_proc.kill()
                await bus_proc.wait()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
