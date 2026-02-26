# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Integration tests for openclaw-agent-server binary.

Starts the real TypeScript binary, sends requests via HTTP, and verifies the
full pipeline: binary boots → accepts a turn → calls the LLM → tool output
returned via SSE.

TestOpenClawRunner: basic tests without AgentBus.
TestOpenClawAgentBus: tests with a real AgentBus server for safety integration.

Run with env vars:
  OPENCLAW_RUNNER_BIN=<path> AGENT_BUS_SERVER_BIN=<path> pytest test_openclaw_runner.py
"""

import asyncio
import json
import os
import socket
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio

HOST = "127.0.0.1"

# ── Helpers ──────────────────────────────────────────────────────────


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _read_api_key() -> str | None:
    key = os.environ.get("LLM_API_KEY")
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


def _resolve_binary() -> str | None:
    """Resolve openclaw-agent-server binary from OPENCLAW_RUNNER_BIN env var."""
    bin_path = os.environ.get("OPENCLAW_RUNNER_BIN")
    if bin_path and os.path.isfile(bin_path):
        return bin_path
    return None


def _url(port: int, path: str) -> str:
    return f"http://{HOST}:{port}{path}"


async def _collect_sse(resp: aiohttp.ClientResponse) -> list[dict]:
    """Read all SSE data chunks from an aiohttp response."""
    chunks = []
    async for line in resp.content:
        decoded = line.decode().strip()
        if decoded.startswith("data: "):
            chunks.append(json.loads(decoded[6:]))
    return chunks


async def _start_process(binary, args, env, tmp_path, name):
    """Start a subprocess with stdout/stderr logging. Returns (process, logs_dir)."""
    stdout_log = tmp_path / f"{name}_stdout.log"
    stderr_log = tmp_path / f"{name}_stderr.log"
    stdout_f = open(stdout_log, "w")
    stderr_f = open(stderr_log, "w")

    process = await asyncio.create_subprocess_exec(
        binary,
        *args,
        stdout=stdout_f,
        stderr=stderr_f,
        env=env,
    )
    return process, stdout_f, stderr_f


async def _wait_for_http(port, path, process, tmp_path, name, timeout=30):
    """Poll an HTTP endpoint until ready. Fails if process exits or times out."""
    stdout_log = tmp_path / f"{name}_stdout.log"
    stderr_log = tmp_path / f"{name}_stderr.log"
    deadline = asyncio.get_event_loop().time() + timeout

    async with aiohttp.ClientSession() as session:
        while asyncio.get_event_loop().time() < deadline:
            if process.returncode is not None:
                pytest.fail(
                    f"{name} exited with code {process.returncode}\n"
                    f"stdout: {stdout_log.read_text()}\n"
                    f"stderr: {stderr_log.read_text()}"
                )
            try:
                async with session.get(
                    _url(port, path),
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    if resp.status == 200:
                        return
            except (aiohttp.ClientError, ConnectionRefusedError, OSError):
                pass
            await asyncio.sleep(0.3)

    process.terminate()
    await process.wait()
    pytest.fail(
        f"{name} failed to start on port {port} within {timeout}s\n"
        f"stdout: {stdout_log.read_text()}\n"
        f"stderr: {stderr_log.read_text()}"
    )


async def _kill_process(process, *file_handles):
    """Terminate a process and close file handles."""
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
    for f in file_handles:
        f.close()


# ── Skip conditions ──────────────────────────────────────────────────

_API_KEY = _read_api_key()
_BINARY = _resolve_binary()
_BUS_BINARY = os.environ.get("AGENT_BUS_SERVER_BIN")

skip_no_api_key = pytest.mark.skipif(
    _API_KEY is None,
    reason="No LLM_API_KEY found (env or ~/.agentkernel/config)",
)
skip_no_binary = pytest.mark.skipif(
    _BINARY is None,
    reason="OPENCLAW_RUNNER_BIN not set (build openclaw_runner or set env var)",
)
skip_no_bus_binary = pytest.mark.skipif(
    not _BUS_BINARY,
    reason="AGENT_BUS_SERVER_BIN not set",
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def openclaw_server(tmp_path: Path):
    """Start the openclaw-agent-server binary, yield (port, config, process)."""
    port = _find_free_port()

    config = {
        "agent_id": "openclaw-test-agent",
        "nonce": "test-nonce-12345",
        "http_port": port,
        "system_prompt": (
            "You are a helpful assistant. When asked to run a shell command, "
            "use the bash tool with the exact command given. Return only the "
            "tool output, no extra commentary."
        ),
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "tools": ["bash"],
        "thinking_level": "none",
        "api_key": _API_KEY,
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    stdout_f = open(stdout_log, "w")
    stderr_f = open(stderr_log, "w")

    env = os.environ.copy()

    process = await asyncio.create_subprocess_exec(
        _BINARY,
        "--config",
        str(config_path),
        stdout=stdout_f,
        stderr=stderr_f,
        env=env,
    )

    # Poll /health until ready (timeout 30s)
    deadline = asyncio.get_event_loop().time() + 30
    ready = False
    async with aiohttp.ClientSession() as session:
        while asyncio.get_event_loop().time() < deadline:
            if process.returncode is not None:
                stdout_f.close()
                stderr_f.close()
                pytest.fail(
                    f"Binary exited with code {process.returncode}\n"
                    f"stdout: {stdout_log.read_text()}\n"
                    f"stderr: {stderr_log.read_text()}"
                )
            try:
                async with session.get(
                    _url(port, "/health"),
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except (aiohttp.ClientError, ConnectionRefusedError, OSError):
                pass
            await asyncio.sleep(0.3)

    if not ready:
        process.terminate()
        await process.wait()
        stdout_f.close()
        stderr_f.close()
        pytest.fail(
            f"Binary failed to start on port {port} within 30s\n"
            f"stdout: {stdout_log.read_text()}\n"
            f"stderr: {stderr_log.read_text()}"
        )

    yield port, config, process, tmp_path

    # Teardown
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
    stdout_f.close()
    stderr_f.close()


@pytest_asyncio.fixture
async def openclaw_with_agentbus(tmp_path: Path):
    """Start agent_bus_server + openclaw-agent-server wired together.

    The bus runs with --run-decider (OnByDefault policy = auto-approve,
    no voter = no rejections). Yields (openclaw_port, bus_port, config, tmp_path).
    """
    bus_port = _find_free_port()
    openclaw_port = _find_free_port()
    bus_id = "openclaw-test-agent"

    env = os.environ.copy()
    env["RUST_LOG"] = "info"

    # 1. Start AgentBus server with auto-approve decider
    bus_proc, bus_stdout, bus_stderr = await _start_process(
        _BUS_BINARY,
        ["--port", str(bus_port), "--run-decider", bus_id],
        env,
        tmp_path,
        "bus",
    )

    # Give the bus server a moment to bind its port
    await asyncio.sleep(1.0)
    if bus_proc.returncode is not None:
        bus_stdout.close()
        bus_stderr.close()
        pytest.fail(
            f"agent_bus_server exited with code {bus_proc.returncode}\n"
            f"stderr: {(tmp_path / 'bus_stderr.log').read_text()}"
        )

    # 2. Start openclaw-agent-server with agentbus_url pointing to the bus
    config = {
        "agent_id": bus_id,
        "nonce": "test-nonce-bus",
        "http_port": openclaw_port,
        "system_prompt": (
            "You are a helpful assistant. When asked to run a shell command, "
            "use the bash tool with the exact command given. Return only the "
            "tool output, no extra commentary."
        ),
        "model": "claude-sonnet-4-5",
        "provider": "anthropic",
        "tools": ["bash"],
        "thinking_level": "none",
        "api_key": _API_KEY,
        "agentbus_url": f"memory://{bus_port}",
        "disable_safety": False,
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    openclaw_proc, openclaw_stdout, openclaw_stderr = await _start_process(
        _BINARY,
        ["--config", str(config_path)],
        env,
        tmp_path,
        "openclaw",
    )

    await _wait_for_http(openclaw_port, "/health", openclaw_proc, tmp_path, "openclaw")

    yield openclaw_port, bus_port, config, tmp_path

    # Teardown both processes
    await _kill_process(openclaw_proc, openclaw_stdout, openclaw_stderr)
    await _kill_process(bus_proc, bus_stdout, bus_stderr)


# ── Tests ────────────────────────────────────────────────────────────


@skip_no_api_key
@skip_no_binary
@pytest.mark.timeout(180)
class TestOpenClawRunner:
    @pytest.mark.asyncio
    async def test_health(self, openclaw_server):
        """GET /health returns ok with the correct agent_id."""
        port, config, _, _ = openclaw_server

        async with aiohttp.ClientSession() as session:
            async with session.get(_url(port, "/health")) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["agent_id"] == config["agent_id"]

    @pytest.mark.asyncio
    async def test_turn_completes(self, openclaw_server):
        """POST /v1/turn returns an SSE stream that terminates with done:true."""
        port, config, _, tmp_path = openclaw_server

        payload = {
            "agent_id": config["agent_id"],
            "nonce": config["nonce"],
            "body": json.dumps(
                {"messages": [{"role": "user", "content": "Say hello"}]}
            ),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                _url(port, "/v1/turn"),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                assert resp.status == 200
                chunks = await _collect_sse(resp)

        # Diagnostic: dump chunks and logs on failure
        stderr_text = (tmp_path / "stderr.log").read_text()
        diag = (
            f"chunks={json.dumps(chunks, indent=2)}\nstderr.log:\n{stderr_text[-2000:]}"
        )

        assert len(chunks) >= 1, f"No SSE chunks received.\n{diag}"
        assert chunks[-1]["done"] is True, f"Last chunk not done.\n{diag}"

        errors = [c["error"] for c in chunks if c.get("error")]
        assert not errors, f"Turn errors: {errors}\n{diag}"

        body = "".join(c.get("body", "") for c in chunks)
        assert len(body) > 0, f"Empty response body.\n{diag}"

    @pytest.mark.asyncio
    async def test_tool_call(self, openclaw_server):
        """POST /v1/turn with a shell command, verify tool output in response."""
        port, config, _, tmp_path = openclaw_server

        payload = {
            "agent_id": config["agent_id"],
            "nonce": config["nonce"],
            "body": json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Run this shell command and return only its output: "
                                "echo hello_from_openclaw"
                            ),
                        }
                    ]
                }
            ),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                _url(port, "/v1/turn"),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                assert resp.status == 200
                chunks = await _collect_sse(resp)

        stderr_text = (tmp_path / "stderr.log").read_text()
        diag = (
            f"chunks={json.dumps(chunks, indent=2)}\nstderr.log:\n{stderr_text[-2000:]}"
        )

        assert len(chunks) >= 1, f"No SSE chunks.\n{diag}"
        assert chunks[-1]["done"] is True, f"Last chunk not done.\n{diag}"

        errors = [c["error"] for c in chunks if c.get("error")]
        assert not errors, f"Turn errors: {errors}\n{diag}"

        # Check SSE body and history for tool output
        body = "".join(c.get("body", "") for c in chunks)

        async with aiohttp.ClientSession() as session:
            async with session.post(_url(port, "/v1/history"), json={}) as resp:
                history = (await resp.json())["entries"]

        assistant_text = " ".join(
            e["content"] for e in history if e["role"] == "assistant"
        )
        all_text = body + " " + assistant_text

        assert "hello_from_openclaw" in all_text, (
            f"Expected 'hello_from_openclaw' in response.\n"
            f"SSE body: {body!r}\n"
            f"History: {history!r}\n{diag}"
        )

    @pytest.mark.asyncio
    async def test_history(self, openclaw_server):
        """After a turn, /v1/history has at least the user message."""
        port, config, _, _ = openclaw_server

        payload = {
            "agent_id": config["agent_id"],
            "nonce": config["nonce"],
            "body": json.dumps(
                {"messages": [{"role": "user", "content": "Say hello"}]}
            ),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                _url(port, "/v1/turn"),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                await _collect_sse(resp)

            async with session.post(_url(port, "/v1/history"), json={}) as resp:
                assert resp.status == 200
                data = await resp.json()
                entries = data["entries"]
                assert len(entries) >= 1
                assert any(e["role"] == "user" for e in entries)


# ── AgentBus integration tests ───────────────────────────────────────


@skip_no_api_key
@skip_no_binary
@skip_no_bus_binary
@pytest.mark.timeout(180)
class TestOpenClawAgentBus:
    """E2E tests: openclaw-agent-server wired to a real AgentBus server.

    The bus runs with --run-decider and OnByDefault policy (auto-approve).
    No voter is started, so all intentions are committed immediately.
    """

    @pytest.mark.asyncio
    async def test_tool_approved_by_agentbus(self, openclaw_with_agentbus):
        """Tool call goes through the full safety pipeline and executes.

        AgentBus decider auto-approves (OnByDefault, no voter), so the
        tool should execute normally. Verifies the openclaw binary correctly:
          1. Proposes the intention to AgentBus
          2. Polls and receives a Commit
          3. Executes the tool
          4. Returns the tool output to the LLM
        """
        openclaw_port, bus_port, config, tmp_path = openclaw_with_agentbus

        payload = {
            "agent_id": config["agent_id"],
            "nonce": config["nonce"],
            "body": json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Run this shell command and return only its output: "
                                "echo agentbus_approved"
                            ),
                        }
                    ]
                }
            ),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                _url(openclaw_port, "/v1/turn"),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                assert resp.status == 200
                chunks = await _collect_sse(resp)

        openclaw_stderr = (tmp_path / "openclaw_stderr.log").read_text()
        bus_stderr = (tmp_path / "bus_stderr.log").read_text()
        diag = (
            f"chunks={json.dumps(chunks, indent=2)}\n"
            f"openclaw stderr (last 2000):\n{openclaw_stderr[-2000:]}\n"
            f"bus stderr (last 2000):\n{bus_stderr[-2000:]}"
        )

        assert len(chunks) >= 1, f"No SSE chunks.\n{diag}"
        assert chunks[-1]["done"] is True, f"Last chunk not done.\n{diag}"

        errors = [c["error"] for c in chunks if c.get("error")]
        assert not errors, f"Turn errors: {errors}\n{diag}"

        # The tool should have executed — check output
        body = "".join(c.get("body", "") for c in chunks)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                _url(openclaw_port, "/v1/history"), json={}
            ) as resp:
                history = (await resp.json())["entries"]

        assistant_text = " ".join(
            e["content"] for e in history if e["role"] == "assistant"
        )
        all_text = body + " " + assistant_text

        assert "agentbus_approved" in all_text, (
            f"Expected 'agentbus_approved' in response — tool should have "
            f"executed after AgentBus approved it.\n"
            f"SSE body: {body!r}\n"
            f"History: {history!r}\n{diag}"
        )

        # Verify openclaw logged that it connected to AgentBus
        assert "AgentBus" in openclaw_stderr or "agentbus" in openclaw_stderr, (
            f"Expected openclaw to log AgentBus connection.\n{diag}"
        )
