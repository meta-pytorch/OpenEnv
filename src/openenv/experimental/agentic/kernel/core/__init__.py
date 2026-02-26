# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""AgentKernel - main entry point for the agentkernel system."""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _add_no_proxy(hosts: list[str]) -> None:
    """Append *hosts* to NO_PROXY (both cases) if not already present."""
    existing = os.environ.get("NO_PROXY", "")
    entries = [e.strip() for e in existing.split(",") if e.strip()]
    for host in hosts:
        if host not in entries:
            entries.append(host)
    merged = ",".join(entries)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _extract_kubeconfig_host(kubeconfig_path: str) -> str | None:
    """Extract the API server host from a kubeconfig file."""
    try:
        from urllib.parse import urlparse

        import yaml

        with open(os.path.expanduser(kubeconfig_path)) as f:
            cfg = yaml.safe_load(f)
        server = cfg.get("clusters", [{}])[0].get("cluster", {}).get("server", "")
        parsed = urlparse(server)
        host = parsed.hostname or ""
        # For IPv6 addresses like [2401:db00:...], urlparse strips the brackets
        return host if host else None
    except Exception:
        return None


from .backends.agent_client import AgentClient
from .backends.bwrap.spawner import BwrapSpawnerService
from .backends.local.packaging import LocalPackagingService
from .backends.local.resolver import LocalResolver
from .backends.local.spawner import LocalSpawnerService
from .backends.ports import PortAllocator
from .bus import BusService
from .config import (  # noqa: F401
    Agent,
    AgentBusConfig,
    AgentImage,
    AgentState,
    CreateTeamRequest,
    PackageJob,
    SourceBundle,
    SpawnRequest,
    SpawnResult,
    TransportFormat,
    TurnRequest,
    TurnResponse,
)
from .plugin import AgentTypePlugin  # noqa: F401
from .storage.blob import LocalBlobStore
from .storage.images import ImageStore
from .storage.registry import AgentRegistry
from .storage.uri import URIDownloader


class AgentKernel:
    """Main entry point for the agentkernel system.

    Wires together storage, services, and the spawner into a single
    facade. Use this class to create teams, build images, spawn
    agents, and communicate with them.

    Args:
        backend: "local" for no isolation, "bwrap" for sandbox,
            "kubernetes" for k8s cluster.
        base_dir: Root directory for all kernel data.
        plugins: List of AgentTypePlugin instances. Each plugin
            handles spawn config and launch for one agent type.
            If None, the kernel works but spawn() will raise on
            any agent type. The **caller** decides which plugins
            to register.
        **kwargs: Backend-specific options.  For kubernetes:
            namespace, base_image, kubeconfig, registry_url.
            For S3 blob storage: s3_bucket, s3_prefix.
    """

    @classmethod
    def from_config(cls, config_path: str | Path, **overrides: Any) -> "AgentKernel":
        """Create an AgentKernel from a YAML config file.

        The file should contain ``backend`` and any backend-specific keys
        (e.g. ``namespace``, ``base_image`` for kubernetes).
        ``base_dir`` is optional (defaults to /tmp/agentkernel).
        Extra **overrides are merged on top of the file contents.
        """
        import yaml

        path = Path(config_path).expanduser()
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        cfg.update(overrides)

        backend = cfg.pop("backend", "local")
        base_dir = Path(cfg.pop("base_dir", "/tmp/agentkernel"))
        return cls(backend=backend, base_dir=base_dir, **cfg)

    def __init__(
        self,
        backend: str = "local",
        base_dir: Path | None = None,
        plugins: list[AgentTypePlugin] | None = None,
        **kwargs: Any,
    ) -> None:
        if base_dir is None:
            base_dir = Path("/tmp/agentkernel")
        logger.info(
            "Initializing AgentKernel backend=%s base_dir=%s", backend, base_dir
        )
        self._backend = backend
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Build plugins dict from list
        plugins_dict: dict[str, AgentTypePlugin] = {}
        for p in plugins or []:
            plugins_dict[p.agent_type] = p

        # Storage
        self.blob_store = LocalBlobStore(base_dir / "blobs")
        self.image_store = ImageStore(base_dir / "images")
        self.agent_registry = AgentRegistry()

        # Bus service (read-only access to agent bus entries)
        self.bus = BusService(self.agent_registry)

        # S3 blob store (optional)
        s3_bucket = kwargs.get("s3_bucket")
        s3_prefix = kwargs.get("s3_prefix", "blobs")
        self.s3_blob_store = None
        if s3_bucket:
            from .storage.s3_blob import S3BlobStore

            self.s3_blob_store = S3BlobStore(bucket=s3_bucket, prefix=s3_prefix)
            logger.info("S3 blob store enabled: s3://%s/%s", s3_bucket, s3_prefix)

        # URIDownloader (shared across packaging services)
        downloader = URIDownloader(blob_store=self.blob_store)

        # Infrastructure
        port_allocator = PortAllocator()
        workspaces_dir = base_dir / "workspaces"

        # Spawner + packaging + agent client (backend-specific)
        runner_binary = kwargs.get("runner_binary")

        if backend == "local":
            self.spawner = LocalSpawnerService(
                self.agent_registry,
                self.image_store,
                workspaces_dir,
                port_allocator,
                runner_binary=runner_binary,
                plugins=plugins_dict,
            )
            self.packaging = LocalPackagingService(downloader, self.image_store)
            self.agent_client = AgentClient(LocalResolver(self.agent_registry))
        elif backend == "bwrap":
            self.spawner = BwrapSpawnerService(
                self.agent_registry,
                self.image_store,
                workspaces_dir,
                port_allocator,
                plugins=plugins_dict,
            )
            self.packaging = LocalPackagingService(downloader, self.image_store)
            self.agent_client = AgentClient(LocalResolver(self.agent_registry))
        elif backend == "kubernetes":
            from .backends.kubernetes.packaging import OciPackagingService
            from .backends.kubernetes.resolver import KubernetesResolver
            from .backends.kubernetes.spawner import KubernetesSpawnerService

            namespace = kwargs.get("namespace", "default")
            base_image = kwargs.get("base_image", "agentkernel:latest")
            kubeconfig = kwargs.get("kubeconfig")
            registry_url = kwargs.get("registry_url", "")
            debug = kwargs.get("debug", False)

            # Add the k8s API server to NO_PROXY so it bypasses any
            # HTTP proxy (especially important for bare IPv6 addresses
            # that won't match wildcard domain patterns).
            if kubeconfig:
                api_host = _extract_kubeconfig_host(kubeconfig)
                if api_host:
                    _add_no_proxy([api_host])

            self.spawner = KubernetesSpawnerService(
                agent_registry=self.agent_registry,
                image_store=self.image_store,
                namespace=namespace,
                base_image=base_image,
                port_allocator=port_allocator,
                kubeconfig=kubeconfig,
                debug=debug,
                plugins=plugins_dict,
            )
            self.packaging = OciPackagingService(
                registry_url=registry_url,
                base_image=base_image,
                downloader=downloader,
                image_store=self.image_store,
            )
            self.agent_client = AgentClient(KubernetesResolver(self.agent_registry))
        else:
            raise ValueError(f"Unknown backend: {backend}")

    async def status(
        self,
        team_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return status of all agents, enriched with live backend checks.

        Each entry contains:
            agent_id, name, team_id, state (from registry), image_id,
            http_port, pid, created_at, metadata,
            live (bool from spawner.is_running()),
            and backend-specific fields:
                - local/bwrap: process_alive (bool)
                - kubernetes: pod_phase, port_forward (alive/dead/n-a)
        """
        agents = await self.agent_registry.list(team_id=team_id)
        results = []

        for agent in agents:
            entry: dict[str, Any] = {
                "agent_id": agent.id,
                "name": agent.name,
                "team_id": agent.team_id,
                "state": agent.state.value,
                "image_id": agent.image_id,
                "http_port": agent.http_port,
                "pid": agent.pid,
                "created_at": agent.created_at.isoformat(),
                "metadata": dict(agent.metadata),
            }

            # Live check via spawner
            try:
                entry["live"] = await self.spawner.is_running(agent.id)
            except Exception:
                entry["live"] = None

            # Backend-specific enrichment
            if self._backend in ("local", "bwrap"):
                procs = getattr(self.spawner, "_processes", {})
                proc = procs.get(agent.id)
                entry["process_alive"] = proc is not None and proc.returncode is None

            elif self._backend == "kubernetes":
                # Pod phase
                try:
                    from .backends.kubernetes.spawner import KubernetesSpawnerService

                    spawner: KubernetesSpawnerService = self.spawner  # type: ignore[assignment]
                    import asyncio

                    rname = spawner._resource_name(agent.id)
                    core = spawner._get_core_api()
                    pod = await asyncio.to_thread(
                        core.read_namespaced_pod, rname, spawner._namespace
                    )
                    entry["pod_phase"] = pod.status.phase

                    # Container status details (for debugging failures)
                    statuses = pod.status.container_statuses or []
                    if statuses:
                        cs = statuses[0]
                        if cs.state.waiting:
                            entry["pod_reason"] = cs.state.waiting.reason
                        elif cs.state.terminated:
                            entry["pod_reason"] = cs.state.terminated.reason
                            entry["exit_code"] = cs.state.terminated.exit_code
                except Exception as e:
                    entry["pod_phase"] = f"error: {e}"

                # Port-forward status
                pf_procs = getattr(self.spawner, "_port_forwards", {})
                pf = pf_procs.get(agent.id)
                if pf is None:
                    entry["port_forward"] = "none"
                elif pf.returncode is None:
                    entry["port_forward"] = "alive"
                else:
                    entry["port_forward"] = f"dead (rc={pf.returncode})"

            results.append(entry)

        return results

    async def cleanup(self) -> None:
        """Kill all agents and clean up resources."""
        agents = await self.agent_registry.list()
        active = [
            a
            for a in agents
            if a.state in (AgentState.RUNNING, AgentState.STARTING, AgentState.IDLE)
        ]
        if active:
            logger.info("Cleaning up %d active agent(s)", len(active))
        for agent in active:
            try:
                await self.spawner.kill(agent.id)
            except Exception:
                logger.warning("Failed to kill agent %s", agent.id, exc_info=True)
