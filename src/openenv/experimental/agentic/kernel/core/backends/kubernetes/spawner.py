# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Kubernetes spawner service - runs agents as pods in a k8s cluster."""

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from typing import Any

import aiohttp

from ...config import (
    Agent,
    AgentBusConfig,
    AgentState,
    CreateTeamRequest,
    SpawnRequest,
    SpawnResult,
)
from ...plugin import AgentTypePlugin
from ...storage.images import ImageStore
from ...storage.registry import AgentRegistry
from ..ports import PortAllocator

logger = logging.getLogger(__name__)

# Fixed port inside the container — the runner always listens here.
_POD_PORT = 9000


class KubernetesSpawnerService:
    """SpawnerService for Kubernetes clusters.

    Spawns agents as Pods with ClusterIP Services, then tunnels traffic
    via ``kubectl port-forward`` so the local AgentClient can reach them
    at ``http://127.0.0.1:{local_port}``.

    Config is delivered to each Pod via a ConfigMap mounted at
    ``/workspace/config.json``.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        image_store: ImageStore,
        namespace: str,
        base_image: str,
        port_allocator: PortAllocator,
        kubeconfig: str | None = None,
        debug: bool = False,
        plugins: dict[str, AgentTypePlugin] | None = None,
    ) -> None:
        self._agents = agent_registry
        self._image_store = image_store
        self._namespace = namespace
        self._base_image = base_image
        self._port_allocator = port_allocator
        self._kubeconfig = os.path.expanduser(kubeconfig) if kubeconfig else None
        self._debug = debug
        self._plugins = plugins or {}

        # Team capacity tracking (same model as local/bwrap)
        self._teams: dict[str, dict[str, int]] = {}
        self._team_used: dict[str, dict[str, int]] = {}

        # Port-forward subprocesses keyed by agent_id
        self._port_forwards: dict[str, asyncio.subprocess.Process] = {}

        # Lazy-loaded k8s client
        self._core_api: Any | None = None

    def _get_core_api(self) -> Any:
        """Return (and lazily create) the k8s CoreV1Api client."""
        if self._core_api is None:
            import kubernetes  # pyre-ignore[21]

            if self._kubeconfig:
                kubernetes.config.load_kube_config(config_file=self._kubeconfig)
            else:
                try:
                    kubernetes.config.load_incluster_config()
                except kubernetes.config.ConfigException:  # pyre-ignore[66]
                    kubernetes.config.load_kube_config()

            # Ensure the kubernetes client talks directly to the API
            # server and never routes through an HTTP proxy. The
            # client reads HTTPS_PROXY at pool-creation time and its
            # urllib3 ProxyManager doesn't honour NO_PROXY for bare
            # IP addresses.
            cfg = kubernetes.client.Configuration.get_default_copy()
            cfg.proxy = None
            self._core_api = kubernetes.client.CoreV1Api(
                kubernetes.client.ApiClient(cfg)
            )
        return self._core_api  # type: ignore[return-value]

    @staticmethod
    def _resource_name(agent_id: str) -> str:
        """Deterministic k8s resource name from agent ID."""
        return f"agent-{agent_id[:8]}"

    # ── Team Management ───────────────────────────────────────────────

    async def create_team(self, request: CreateTeamRequest) -> None:
        """Reserve capacity for a team (tracked in-memory)."""
        if request.team_id in self._teams:
            raise KeyError(f"Team already exists: {request.team_id}")
        self._teams[request.team_id] = dict(request.resources)
        self._team_used[request.team_id] = {k: 0 for k in request.resources}

    async def delete_team(self, team_id: str) -> None:
        """Delete a team, killing all its agents first."""
        if team_id not in self._teams:
            raise KeyError(f"Team not found: {team_id}")

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

    # ── Spawn ─────────────────────────────────────────────────────────

    async def spawn(self, request: SpawnRequest) -> SpawnResult:
        """Create a Pod + Service for the agent and set up port-forwarding."""
        import kubernetes

        # Enforce team capacity
        capacity_incremented = False
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
            capacity_incremented = True

        # Resolve agentbus config: allocate port for in-process buses
        resolved_agentbus = request.agentbus
        if resolved_agentbus:
            from ...bus_config import parse_agentbus_url

            parsed = parse_agentbus_url(resolved_agentbus.url)
            if parsed.get("host") is None and parsed.get("port") is None:
                bus_port = await self._port_allocator.allocate()
                resolved_agentbus = AgentBusConfig(
                    url=f"memory://{bus_port}",
                    disable_safety=resolved_agentbus.disable_safety,
                )

        # Create agent record
        agent = Agent(
            id=str(uuid.uuid4()),
            name=request.name,
            team_id=request.team_id,
            agent_type=request.agent_type,
            image_id=request.image_id or "",
            spawn_info=request.spawn_info,
            metadata=request.metadata,
            agentbus=resolved_agentbus,
        )
        await self._agents.register(agent)

        rname = self._resource_name(agent.id)
        labels = {
            "agentkernel/agent-id": agent.id,
            "agentkernel/team-id": agent.team_id,
            "app": rname,
        }

        # Build config dict via plugin
        nonce = secrets.token_urlsafe(16)

        plugin = self._plugins.get(request.agent_type)
        if not plugin:
            raise ValueError(
                f"Unknown agent type: {request.agent_type}. "
                f"Available: {list(self._plugins)}"
            )

        # Use a workspace_path for config building (k8s uses /workspace)
        from pathlib import Path

        workspace_path = Path("/workspace")
        config = plugin.build_config(agent, nonce, _POD_PORT, workspace_path, request)

        # Override scratch_directory for container path
        config["scratch_directory"] = "/workspace/scratch"

        # Append cross-cutting agentbus fields
        if agent.agentbus:
            config["agentbus_url"] = agent.agentbus.url
            config["disable_safety"] = agent.agentbus.disable_safety

        # Resolve launch command from plugin
        agent_cmd = plugin.resolve_command()

        core = self._get_core_api()
        local_port = None

        try:
            # 1. ConfigMap
            configmap = kubernetes.client.V1ConfigMap(
                metadata=kubernetes.client.V1ObjectMeta(
                    name=rname,
                    namespace=self._namespace,
                    labels=labels,
                ),
                data={"config.json": json.dumps(config, indent=2)},
            )
            await asyncio.to_thread(
                core.create_namespaced_config_map, self._namespace, configmap
            )

            # 2. Pod
            env_vars = [
                kubernetes.client.V1EnvVar(name="AGENT_ID", value=agent.id),
                kubernetes.client.V1EnvVar(
                    name="AGENT_HTTP_PORT", value=str(_POD_PORT)
                ),
                kubernetes.client.V1EnvVar(
                    name="AGENT_CONFIG_PATH", value="/workspace/config.json"
                ),
            ]
            # Forward host env vars
            for env_name in (
                "LLM_API_KEY",
                "OPENAI_API_KEY",
                "AGENTKERNEL_LOG_LEVEL",
            ):
                val = os.environ.get(env_name)
                if val:
                    env_vars.append(
                        kubernetes.client.V1EnvVar(name=env_name, value=val)
                    )
            # Forward request.env
            for k, v in request.env.items():
                env_vars.append(kubernetes.client.V1EnvVar(name=k, value=v))

            # Resolve image_id → registry tag via image store, fall back to base
            if request.image_id:
                stored = self._image_store.get(request.image_id)
                if not stored:
                    raise KeyError(f"Image not found: {request.image_id}")
                image = str(stored.path)
            else:
                image = self._base_image

            container = kubernetes.client.V1Container(
                name="agent",
                image=image,
                command=agent_cmd + ["--config", "/workspace/config.json"],
                ports=[kubernetes.client.V1ContainerPort(container_port=_POD_PORT)],
                env=env_vars,
                volume_mounts=[
                    kubernetes.client.V1VolumeMount(
                        name="config",
                        mount_path="/workspace/config.json",
                        sub_path="config.json",
                    ),
                    kubernetes.client.V1VolumeMount(
                        name="scratch",
                        mount_path="/workspace/scratch",
                    ),
                ],
                readiness_probe=kubernetes.client.V1Probe(
                    http_get=kubernetes.client.V1HTTPGetAction(
                        path="/health", port=_POD_PORT
                    ),
                    initial_delay_seconds=2,
                    period_seconds=2,
                ),
            )

            pod = kubernetes.client.V1Pod(
                metadata=kubernetes.client.V1ObjectMeta(
                    name=rname,
                    namespace=self._namespace,
                    labels=labels,
                ),
                spec=kubernetes.client.V1PodSpec(
                    containers=[container],
                    restart_policy="Never",
                    volumes=[
                        kubernetes.client.V1Volume(
                            name="config",
                            config_map=kubernetes.client.V1ConfigMapVolumeSource(
                                name=rname,
                            ),
                        ),
                        kubernetes.client.V1Volume(
                            name="scratch",
                            empty_dir=kubernetes.client.V1EmptyDirVolumeSource(),
                        ),
                    ],
                ),
            )
            await asyncio.to_thread(core.create_namespaced_pod, self._namespace, pod)

            # 3. ClusterIP Service
            service = kubernetes.client.V1Service(
                metadata=kubernetes.client.V1ObjectMeta(
                    name=rname,
                    namespace=self._namespace,
                    labels=labels,
                ),
                spec=kubernetes.client.V1ServiceSpec(
                    selector={"app": rname},
                    ports=[
                        kubernetes.client.V1ServicePort(
                            port=_POD_PORT, target_port=_POD_PORT
                        )
                    ],
                    type="ClusterIP",
                ),
            )
            await asyncio.to_thread(
                core.create_namespaced_service, self._namespace, service
            )

            # 4. Wait for Pod ready
            logger.info("Waiting for pod %s in namespace %s", rname, self._namespace)
            await self._wait_for_pod_ready(rname, timeout=120.0)

            # 5. Port-forward
            local_port = await self._port_allocator.allocate()
            await self._start_port_forward(agent.id, rname, local_port)
            await self._wait_for_local_ready(local_port, timeout=30.0)

            agent.http_port = local_port
            agent.state = AgentState.RUNNING
            logger.info(
                "Agent %s ready (pod=%s local_port=%d)",
                agent.id,
                rname,
                local_port,
            )

        except Exception:
            logger.error("Failed to spawn agent %s", agent.id, exc_info=True)

            # Stop port-forward subprocess if it was started
            pf = self._port_forwards.pop(agent.id, None)
            if pf:
                pf.terminate()
                try:
                    await asyncio.wait_for(pf.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pf.kill()

            if self._debug:
                logger.info(
                    "Debug mode: preserving k8s resources for pod %s "
                    "(configmap, service) for inspection",
                    rname,
                )
            else:
                await self._cleanup_k8s_resources(agent.id, rname)

            # Release allocated port
            if local_port is not None:
                await self._port_allocator.release(local_port)

            # Roll back team capacity
            if capacity_incremented:
                self._team_used[request.team_id]["cpu"] = max(
                    0, self._team_used[request.team_id]["cpu"] - 1
                )

            raise

        return SpawnResult(agent=agent, nonce=nonce)

    # ── Kill ──────────────────────────────────────────────────────────

    async def kill(self, agent_id: str) -> None:
        """Delete Pod + Service + ConfigMap and stop port-forward."""
        agent = await self._agents.get(agent_id)
        if not agent:
            raise KeyError(f"Agent not found: {agent_id}")

        rname = self._resource_name(agent_id)
        logger.info("Killing agent %s (pod=%s)", agent_id, rname)

        # Stop port-forward subprocess
        pf = self._port_forwards.pop(agent_id, None)
        if pf:
            pf.terminate()
            try:
                await asyncio.wait_for(pf.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pf.kill()

        # Delete k8s resources (skip in debug mode for inspection)
        if self._debug:
            logger.info(
                "Debug mode: preserving k8s resources for %s — "
                "run 'kubectl delete pod,svc,cm %s -n %s' to clean up",
                agent_id,
                rname,
                self._namespace,
            )
        else:
            await self._cleanup_k8s_resources(agent_id, rname)

        # Release local port
        if agent.http_port is not None:
            await self._port_allocator.release(agent.http_port)

        # Release allocated bus port
        if agent.agentbus:
            from ...bus_config import parse_agentbus_url

            try:
                parsed = parse_agentbus_url(agent.agentbus.url)
                bus_port = parsed.get("port")
                if bus_port is not None and parsed.get("host") is None:
                    await self._port_allocator.release(bus_port)
            except ValueError:
                pass

        # Release team capacity
        if agent.team_id in self._team_used:
            used = self._team_used[agent.team_id]
            if "cpu" in used:
                used["cpu"] = max(0, used["cpu"] - 1)

        agent.state = AgentState.STOPPED

    async def get(self, agent_id: str) -> Agent | None:
        return await self._agents.get(agent_id)

    async def is_running(self, agent_id: str) -> bool:
        """Check pod phase via k8s API."""
        rname = self._resource_name(agent_id)
        try:
            core = self._get_core_api()
            pod = await asyncio.to_thread(
                core.read_namespaced_pod, rname, self._namespace
            )
            return pod.status.phase == "Running"
        except Exception:
            return False

    # ── Helpers ───────────────────────────────────────────────────────

    async def _log_pod_diagnostics(self, name: str) -> None:
        """Log pod status, container details, and logs for debugging."""
        core = self._get_core_api()
        try:
            pod = await asyncio.to_thread(
                core.read_namespaced_pod, name, self._namespace
            )
            phase = pod.status.phase
            logger.error("Pod %s phase: %s", name, phase)
            for cs in pod.status.container_statuses or []:
                if cs.state.waiting:
                    logger.error(
                        "  container %s: waiting (%s: %s)",
                        cs.name,
                        cs.state.waiting.reason,
                        cs.state.waiting.message or "",
                    )
                elif cs.state.terminated:
                    logger.error(
                        "  container %s: terminated (%s, exit_code=%s: %s)",
                        cs.name,
                        cs.state.terminated.reason,
                        cs.state.terminated.exit_code,
                        cs.state.terminated.message or "",
                    )
        except Exception:
            logger.debug("Could not read pod status for %s", name, exc_info=True)

        try:
            logs = await asyncio.to_thread(
                core.read_namespaced_pod_log,
                name,
                self._namespace,
                tail_lines=80,
            )
            logger.error("Pod %s logs:\n%s", name, logs)
        except Exception:
            logger.debug("Could not read pod logs for %s", name, exc_info=True)

    async def _wait_for_pod_ready(self, name: str, timeout: float) -> None:
        """Poll pod status until all containers are ready."""
        core = self._get_core_api()
        start = time.time()
        while time.time() - start < timeout:
            pod = await asyncio.to_thread(
                core.read_namespaced_pod, name, self._namespace
            )
            phase = pod.status.phase
            if phase == "Failed" or phase == "Succeeded":
                await self._log_pod_diagnostics(name)
                raise RuntimeError(f"Pod {name} entered terminal phase: {phase}")

            conditions = pod.status.conditions or []
            for cond in conditions:
                if cond.type == "Ready" and cond.status == "True":
                    return

            await asyncio.sleep(1.0)

        await self._log_pod_diagnostics(name)
        raise TimeoutError(f"Pod {name} not ready after {timeout}s")

    async def _start_port_forward(
        self, agent_id: str, svc_name: str, local_port: int
    ) -> None:
        """Start a kubectl port-forward subprocess."""
        cmd = [
            "kubectl",
            "port-forward",
            f"svc/{svc_name}",
            f"{local_port}:{_POD_PORT}",
            "-n",
            self._namespace,
        ]
        if self._kubeconfig:
            cmd.extend(["--kubeconfig", self._kubeconfig])

        logger.info("Starting port-forward: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._port_forwards[agent_id] = proc

    async def _wait_for_local_ready(self, local_port: int, timeout: float) -> None:
        """Poll localhost until the port-forward is serving."""
        start = time.time()
        async with aiohttp.ClientSession() as session:
            while time.time() - start < timeout:
                try:
                    async with session.get(
                        f"http://127.0.0.1:{local_port}/health",
                        timeout=aiohttp.ClientTimeout(total=2.0),
                    ) as resp:
                        if resp.status == 200:
                            return
                except Exception:
                    pass
                await asyncio.sleep(0.5)

        raise TimeoutError(
            f"Port-forward to localhost:{local_port} not ready after {timeout}s"
        )

    async def _cleanup_k8s_resources(self, agent_id: str, rname: str) -> None:
        """Best-effort deletion of ConfigMap, Pod, and Service."""
        import kubernetes  # pyre-ignore[21]

        core = self._get_core_api()
        for kind, method in [
            ("Service", core.delete_namespaced_service),
            ("Pod", core.delete_namespaced_pod),
            ("ConfigMap", core.delete_namespaced_config_map),
        ]:
            try:
                kwargs = {}
                if kind == "Pod":
                    kwargs["grace_period_seconds"] = 5
                await asyncio.to_thread(method, rname, self._namespace, **kwargs)
            except kubernetes.client.exceptions.ApiException as exc:  # pyre-ignore[66]
                if exc.status != 404:
                    logger.warning("Failed to delete %s %s: %s", kind, rname, exc)
            except Exception:
                logger.warning("Failed to delete %s %s", kind, rname, exc_info=True)
