"""
Subprocess lifecycle management for AWM sub-environments.
Each AWM scenario is a self-contained FastAPI application. This module handles:
- Patching generated code (DB path, FastApiMCP injection)
- Starting / stopping subprocess on a random port
- Proxying MCP tool calls to the subprocess
"""

import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from typing import Any

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.config import LoggerSettings, MCPServerSettings, MCPSettings, Settings


logger = logging.getLogger(__name__)

MCP_READY_TIMEOUT = 60
MCP_READY_POLL_INTERVAL = 0.5


def _get_random_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _patch_env_code(full_code: str, db_path: str, host: str, port: int) -> str:
    """
    Patch the generated FastAPI code:
    1. Replace create_engine() call to point to the session-specific DB.
    2. Inject FastApiMCP mount before uvicorn.run().
    3. Fix uvicorn.run() to use the assigned host/port.
    """
    new_lines = [
        "import warnings",
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
    ]

    sql_path = f"sqlite:///{db_path}"

    for line in full_code.split("\n"):
        if "create_engine(" in line:
            left = line.split("create_engine(")[0]
            line = f"{left}create_engine('{sql_path}', connect_args={{'check_same_thread': False}})"

        if "uvicorn.run(app" in line:
            mcp_inject = textwrap.dedent("""\
                from fastapi_mcp import FastApiMCP
                mcp = FastApiMCP(app)
                mcp.mount_http()
            """)
            for inject_line in mcp_inject.strip().split("\n"):
                new_lines.append(f"    {inject_line}")

            line = f"    uvicorn.run(app, host='{host}', port={port})"

        new_lines.append(line)

    return "\n".join(new_lines)


class ScenarioProcess:
    """Manages a single sub-environment subprocess."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._temp_dir: str | None = None
        self._owns_temp_dir: bool = False
        self._server_py: str | None = None
        self._log_file = None
        self._log_path: str | None = None

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def mcp_url(self) -> str | None:
        if self._port is None:
            return None
        return f"http://127.0.0.1:{self._port}/mcp"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, full_code: str, db_path: str, session_dir: str) -> str:
        """
        Start the sub-environment subprocess.

        Args:
            full_code: Generated FastAPI Python code from gen_envs.jsonl.
            db_path: Path to the session-specific SQLite database.
            session_dir: Directory for all session artifacts (server.py, server.log, etc.).

        Returns:
            The MCP URL of the started server.

        Raises:
            RuntimeError: If the server fails to start within the timeout.
        """
        self.stop()

        self._temp_dir = session_dir
        self._owns_temp_dir = False  # caller owns the directory
        self._port = _get_random_port()
        host = "127.0.0.1"

        patched_code = _patch_env_code(full_code, db_path, host, self._port)

        self._server_py = f"{self._temp_dir}/server.py"
        with open(self._server_py, "w", encoding="utf-8") as f:
            f.write(patched_code)

        logger.info(f"Starting sub-env on port {self._port} ...")

        # Write stdout/stderr to a log file instead of PIPE to avoid pipe
        # buffer deadlock.  Generated FastAPI code can emit many warnings to
        # stderr; if the pipe buffer fills up the subprocess blocks on write
        # and never reaches uvicorn.run().
        self._log_path = f"{self._temp_dir}/server.log"
        self._log_file = open(self._log_path, "w", encoding="utf-8")

        self._process = subprocess.Popen(
            [sys.executable, self._server_py],
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

        if not self._wait_for_ready():
            failed_port = self._port
            # Read logged output for the error message
            self._log_file.flush()
            try:
                with open(self._log_path, "r", encoding="utf-8") as lf:
                    logged_output = lf.read()
            except OSError:
                logged_output = ""
            self.stop()
            raise RuntimeError(
                f"Sub-env failed to start on port {failed_port} "
                f"(timeout {MCP_READY_TIMEOUT}s).\n"
                f"Output: {logged_output}"
            )

        logger.info(f"Sub-env ready on port {self._port}, log={self._log_path}")
        logger.info(f"[AWM sub-env] ready port={self._port} log={self._log_path}")
        return self.mcp_url

    def _wait_for_ready(self) -> bool:
        """Poll the MCP endpoint until the server is ready."""
        start = time.time()
        while time.time() - start < MCP_READY_TIMEOUT:
            if self._process.poll() is not None:
                return False

            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=1):
                    time.sleep(0.5)
                    return True
            except (ConnectionRefusedError, OSError, socket.timeout):
                pass

            time.sleep(MCP_READY_POLL_INTERVAL)

        return False

    def stop(self) -> None:
        """Stop the subprocess and clean up temp files."""
        if self._process is not None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass

            self._process = None

        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

        if self._owns_temp_dir and self._temp_dir and os.path.isdir(self._temp_dir):
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        self._port = None
        self._server_py = None


_ENV_VARS_TO_SANITIZE = ["ENV"]


def _make_mcp_settings(mcp_url: str) -> Any:
    """
    Create mcp-agent Settings with sanitized environment variables.
    The mcp-agent Settings class (pydantic-settings) reads from env vars.
    Variables like ENV cause parse errors. We temporarily
    remove known-problematic variables during construction.
    """

    saved: dict[str, str] = {}
    for var in _ENV_VARS_TO_SANITIZE:
        if var in os.environ:
            saved[var] = os.environ.pop(var)

    try:
        return Settings(
            execution_engine="asyncio",
            logger=LoggerSettings(
                type="none",
                transports=["none"],
                progress_display=False,
                level="error",
            ),
            mcp=MCPSettings(
                servers={
                    "sub_env": MCPServerSettings(
                        transport="streamable_http", url=mcp_url
                    ),
                }
            ),
        )
    finally:
        for var, val in saved.items():
            os.environ[var] = val


async def call_mcp_tool(
    mcp_url: str, tool_name: str, arguments: dict, timeout: float = 30.0
) -> dict[str, Any]:
    """Call an MCP tool on a running sub-environment via streamable HTTP transport.

    Returns:
        dict with keys: "success" (bool), "result" (Any), "error" (str | None)
    """
    try:
        settings = _make_mcp_settings(mcp_url)
        app = MCPApp(name="_awm_proxy", settings=settings)
        async with app.run():
            agent = Agent(name="_proxy", server_names=["sub_env"])
            async with agent:
                result = await asyncio.wait_for(
                    agent.call_tool(tool_name, arguments),
                    timeout=timeout,
                )

                parts = []
                for c in result.content:
                    if hasattr(c, "text"):
                        parts.append(c.text)
                    else:
                        parts.append(str(c))

                text = "\n".join(parts)
                if result.isError:
                    return {"success": False, "result": None, "error": text}
                return {"success": True, "result": text, "error": None}

    except asyncio.TimeoutError:
        return {
            "success": False,
            "result": None,
            "error": f"Tool call timed out after {timeout}s",
        }
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


async def list_mcp_tools(mcp_url: str, timeout: float = 15.0) -> list[dict]:
    """List available MCP tools from the sub-environment."""
    try:
        settings = _make_mcp_settings(mcp_url)
        app = MCPApp(name="_awm_list", settings=settings)
        async with app.run():
            agent = Agent(name="_lister", server_names=["sub_env"])
            async with agent:
                result = await asyncio.wait_for(agent.list_tools(), timeout=timeout)
                tools = []
                for t in result.tools:
                    tools.append(
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "inputSchema": t.inputSchema or {},
                        }
                    )
                return tools

    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        return []
