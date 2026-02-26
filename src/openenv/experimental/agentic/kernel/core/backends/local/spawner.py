# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Local spawner service - runs agents as subprocesses without isolation."""

import asyncio
import json
import logging
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

import aiohttp

from ...plugin import AgentTypePlugin

logger = logging.getLogger(__name__)

from ...config import (
    Agent,
    AgentBusConfig,
    AgentState,
    CreateTeamRequest,
    SpawnRequest,
    SpawnResult,
)
from ...storage.images import ImageStore
from ...storage.registry import AgentRegistry
from ..ports import PortAllocator


class LocalSpawnerService:
    """Spawns agents as local Python subprocesses.

    No sandboxing or isolation — agents run with the same permissions
    as the parent process. Suitable for local development and testing.

    Owns the full agent lifecycle: team capacity, agent records,
    image resolution, workspace creation, port allocation, and
    process management.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        image_store: ImageStore,
        workspaces_dir: Path,
        port_allocator: PortAllocator,
        runner_binary: str | None = None,
        plugins: dict[str, AgentTypePlugin] | None = None,
    ) -> None:
        self._agents = agent_registry
        self._image_store = image_store
        self._workspaces_dir = workspaces_dir
        self._port_allocator = port_allocator
        self._runner_binary = runner_binary
        self._plugins = plugins or {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._bus_ports: dict[str, int] = {}  # agent_id -> allocated bus port
        self._teams: dict[str, dict[str, int]] = {}  # team_id -> resource budget
        self._team_used: dict[str, dict[str, int]] = {}  # team_id -> used resources

    async def create_team(self, request: CreateTeamRequest) -> None:
        """Reserve capacity for a team. Config is ignored for local."""
        if request.team_id in self._teams:
            raise KeyError(f"Team already exists: {request.team_id}")
        self._teams[request.team_id] = dict(request.resources)
        self._team_used[request.team_id] = {k: 0 for k in request.resources}

    async def delete_team(self, team_id: str) -> None:
        """Delete a team, killing all its agents first."""
        if team_id not in self._teams:
            raise KeyError(f"Team not found: {team_id}")

        # Kill all agents in the team
        agents = await self._agents.list(team_id=team_id)
        for agent in agents:
            if agent.state in (
                AgentState.RUNNING,
                AgentState.STARTING,
                AgentState.IDLE,
            ):
                await self.kill(agent.id)

        del self._teams[team_id]
        del self._team_used[team_id]

    async def spawn(self, request: SpawnRequest) -> SpawnResult:
        """Spawn an agent as a local subprocess."""
        # Enforce team capacity
        if request.team_id and request.team_id not in self._teams:
            raise KeyError(f"Team not found: {request.team_id}")
        if request.team_id in self._teams:
            budget = self._teams[request.team_id]
            used = self._team_used[request.team_id]
            cpu_total = budget.get("cpu", 0)
            cpu_used = used.get("cpu", 0)
            if cpu_used >= cpu_total:
                raise ValueError(
                    f"Team {request.team_id} has no capacity "
                    f"(used {cpu_used}/{cpu_total})"
                )
            self._team_used[request.team_id]["cpu"] = cpu_used + 1

        # Resolve image
        image_id = request.image_id or ""
        if image_id:
            if not self._image_store.exists(image_id):
                raise KeyError(f"Image not found: {image_id}")

        # Resolve agentbus config: allocate port for in-process buses
        resolved_agentbus = request.agentbus
        allocated_bus_port: int | None = None
        if resolved_agentbus:
            from ...bus_config import parse_agentbus_url

            parsed = parse_agentbus_url(resolved_agentbus.url)
            if parsed.get("host") is None and parsed.get("port") is None:
                allocated_bus_port = await self._port_allocator.allocate()
                resolved_agentbus = AgentBusConfig(
                    url="memory://",
                    disable_safety=resolved_agentbus.disable_safety,
                )

        # Create agent record
        agent = Agent(
            id=str(uuid.uuid4()),
            name=request.name,
            team_id=request.team_id,
            agent_type=request.agent_type,
            image_id=image_id,
            spawn_info=request.spawn_info,
            metadata=request.metadata,
            agentbus=resolved_agentbus,
        )
        await self._agents.register(agent)
        if allocated_bus_port is not None:
            self._bus_ports[agent.id] = allocated_bus_port

        # Allocate port
        http_port = await self._port_allocator.allocate()

        # Create workspace
        workspace_path = self._workspaces_dir / agent.id
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Write config for the agent runner.
        nonce = secrets.token_urlsafe(16)

        # Dispatch to plugin
        plugin = self._plugins.get(request.agent_type)
        if not plugin:
            raise ValueError(
                f"Unknown agent type: {request.agent_type}. "
                f"Available: {list(self._plugins)}"
            )
        config = plugin.build_config(agent, nonce, http_port, workspace_path, request)
        cmd = plugin.resolve_command()

        # Append cross-cutting agentbus fields
        if agent.agentbus:
            config["agentbus_url"] = agent.agentbus.url
            config["disable_safety"] = agent.agentbus.disable_safety

        # Inject allocated bus port (memory:// URLs don't carry the port inline)
        if allocated_bus_port is not None:
            config["agentbus_port"] = allocated_bus_port

        config_path = workspace_path / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")

        # Build PYTHONPATH so the child process can import agentkernel
        # and any other source-tree packages the parent has on sys.path.
        existing_pypath = os.environ.get("PYTHONPATH", "")
        extra_paths = [
            p for p in sys.path if p and p not in existing_pypath.split(os.pathsep)
        ]
        merged_pypath = os.pathsep.join(
            extra_paths + ([existing_pypath] if existing_pypath else [])
        )

        agent_env = {
            **os.environ,
            **request.env,
            "AGENT_ID": agent.id,
            "AGENT_HTTP_PORT": str(http_port),
            "AGENT_CONFIG_PATH": str(config_path),
            "PYTHONPATH": merged_pypath,
        }

        try:
            logger.info(
                "Spawning %s agent %s (name=%s) on port %d",
                request.agent_type,
                agent.id,
                agent.name,
                http_port,
            )
            if self._runner_binary:
                cmd = [self._runner_binary, "--config", str(config_path)]
            else:
                cmd = cmd + ["--config", str(config_path)]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=agent_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self._processes[agent.id] = process
            agent.pid = process.pid
            agent.http_port = http_port
            agent.state = AgentState.STARTING

            await self._wait_for_ready(agent, http_port, timeout=30.0)
            agent.state = AgentState.RUNNING
            logger.info(
                "Agent %s ready (pid=%d port=%d)", agent.id, process.pid, http_port
            )
        except Exception:
            logger.error("Failed to spawn agent %s", agent.id, exc_info=True)
            await self._port_allocator.release(http_port)
            raise

        return SpawnResult(agent=agent, nonce=nonce)

    async def kill(self, agent_id: str) -> None:
        """Terminate an agent subprocess and release its resources."""
        agent = await self._agents.get(agent_id)
        if not agent:
            raise KeyError(f"Agent not found: {agent_id}")

        logger.info(
            "Killing agent %s (name=%s pid=%s)", agent_id, agent.name, agent.pid
        )
        if agent.id in self._processes:
            self._processes[agent.id].terminate()
            try:
                await asyncio.wait_for(self._processes[agent.id].wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._processes[agent.id].kill()
            del self._processes[agent.id]

        if agent.http_port is not None:
            await self._port_allocator.release(agent.http_port)

        # Release allocated bus port
        if agent_id in self._bus_ports:
            await self._port_allocator.release(self._bus_ports.pop(agent_id))

        # Release team capacity
        if agent.team_id in self._team_used:
            used = self._team_used[agent.team_id]
            if "cpu" in used:
                used["cpu"] = max(0, used["cpu"] - 1)

        agent.state = AgentState.STOPPED

    async def get(self, agent_id: str) -> Agent | None:
        return await self._agents.get(agent_id)

    async def is_running(self, agent_id: str) -> bool:
        if agent_id not in self._processes:
            return False
        return self._processes[agent_id].returncode is None

    async def _wait_for_ready(
        self, agent: Agent, http_port: int, timeout: float
    ) -> None:
        """Poll /health until the agent is ready."""
        start = time.time()
        async with aiohttp.ClientSession() as session:
            while time.time() - start < timeout:
                proc = self._processes.get(agent.id)
                if proc and proc.returncode is not None:
                    stderr = await proc.stderr.read() if proc.stderr else b""
                    raise RuntimeError(
                        f"Agent {agent.id} died during startup: {stderr.decode()}"
                    )

                try:
                    async with session.get(
                        f"http://127.0.0.1:{http_port}/health",
                        timeout=aiohttp.ClientTimeout(total=1.0),
                    ) as resp:
                        if resp.status == 200:
                            return
                except Exception:
                    pass

                await asyncio.sleep(0.2)

        raise TimeoutError(f"Agent {agent.id} not ready after {timeout}s")
