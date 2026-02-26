# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for the Kubernetes backend (spawner, resolver, packaging).

All k8s API calls and subprocess invocations are mocked so these tests
run without a real cluster or Docker daemon.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentic.kernel.core.backends.kubernetes.packaging import OciPackagingService
from agentic.kernel.core.backends.kubernetes.resolver import KubernetesResolver
from agentic.kernel.core.backends.kubernetes.spawner import KubernetesSpawnerService
from agentic.kernel.core.backends.ports import PortAllocator
from agentic.kernel.core.config import (
    Agent,
    AgentState,
    CreateTeamRequest,
    SourceBundle,
    SpawnRequest,
)
from agentic.kernel.core.storage.blob import LocalBlobStore
from agentic.kernel.core.storage.images import ImageStore
from agentic.kernel.core.storage.registry import AgentRegistry
from agentic.kernel.core.storage.uri import URIDownloader


class _StubPlugin:
    """Minimal plugin for testing spawner mechanics without a real agent type."""

    @property
    def agent_type(self) -> str:
        return "stub"

    def build_config(
        self,
        agent: Agent,
        nonce: str,
        http_port: int,
        workspace_path: Path,
        request: SpawnRequest,
    ) -> dict[str, Any]:
        return {
            "agent_id": agent.id,
            "nonce": nonce,
            "http_port": http_port,
            "scratch_directory": str(workspace_path / "scratch"),
        }

    def resolve_command(self) -> list[str]:
        return [sys.executable, "-m", "http.server"]


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def agent_registry() -> AgentRegistry:
    return AgentRegistry()


@pytest.fixture
def port_allocator() -> PortAllocator:
    return PortAllocator(start=10000, end=10100)


@pytest.fixture
def mock_core_api() -> MagicMock:
    """A mock kubernetes.client.CoreV1Api with all methods used by the spawner."""
    api = MagicMock()
    api.create_namespaced_config_map = MagicMock()
    api.create_namespaced_pod = MagicMock()
    api.create_namespaced_service = MagicMock()
    api.delete_namespaced_config_map = MagicMock()
    api.delete_namespaced_pod = MagicMock()
    api.delete_namespaced_service = MagicMock()

    # read_namespaced_pod returns a pod in Running phase with Ready condition
    ready_cond = SimpleNamespace(type="Ready", status="True")
    api.read_namespaced_pod = MagicMock(
        return_value=SimpleNamespace(
            status=SimpleNamespace(phase="Running", conditions=[ready_cond])
        )
    )
    return api


@pytest.fixture
def spawner(
    agent_registry: AgentRegistry,
    image_store: ImageStore,
    port_allocator: PortAllocator,
    mock_core_api: MagicMock,
) -> KubernetesSpawnerService:
    svc = KubernetesSpawnerService(
        agent_registry=agent_registry,
        image_store=image_store,
        namespace="test-ns",
        base_image="registry.example.com/agentkernel:latest",
        port_allocator=port_allocator,
        plugins={"stub": _StubPlugin()},
    )
    # Inject mock so _get_core_api() skips real k8s config loading
    svc._core_api = mock_core_api
    return svc


@pytest.fixture
def image_store(tmp_path: Path) -> ImageStore:
    return ImageStore(tmp_path / "images")


@pytest.fixture
def blob_store(tmp_path: Path) -> LocalBlobStore:
    return LocalBlobStore(tmp_path / "blobs")


# ── Spawner Tests ─────────────────────────────────────────────────────


class TestKubernetesSpawnerService:
    @pytest.mark.asyncio
    async def test_create_and_delete_team(
        self, spawner: KubernetesSpawnerService
    ) -> None:
        await spawner.create_team(
            CreateTeamRequest(team_id="team-a", resources={"cpu": 2})
        )
        # Duplicate should raise
        with pytest.raises(KeyError):
            await spawner.create_team(
                CreateTeamRequest(team_id="team-a", resources={"cpu": 2})
            )
        await spawner.delete_team("team-a")
        # Deleting again should raise
        with pytest.raises(KeyError):
            await spawner.delete_team("team-a")

    @pytest.mark.asyncio
    async def test_team_capacity_enforcement(
        self, spawner: KubernetesSpawnerService
    ) -> None:
        await spawner.create_team(
            CreateTeamRequest(team_id="cap-team", resources={"cpu": 1})
        )

        # Mock port-forward and health check
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
        ):
            # First spawn should succeed
            result = await spawner.spawn(
                SpawnRequest(name="a1", team_id="cap-team", agent_type="stub")
            )
            assert result.agent.state == AgentState.RUNNING

            # Second spawn should fail (capacity=1)
            with pytest.raises(ValueError, match="no capacity"):
                await spawner.spawn(
                    SpawnRequest(name="a2", team_id="cap-team", agent_type="stub")
                )

    @pytest.mark.asyncio
    async def test_spawn_creates_k8s_resources(
        self,
        spawner: KubernetesSpawnerService,
        mock_core_api: MagicMock,
    ) -> None:
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
        ):
            result = await spawner.spawn(
                SpawnRequest(name="test-agent", agent_type="stub")
            )

        agent = result.agent
        assert agent.state == AgentState.RUNNING
        assert agent.http_port is not None
        assert result.nonce  # nonce is generated

        # Verify k8s API calls
        mock_core_api.create_namespaced_config_map.assert_called_once()
        mock_core_api.create_namespaced_pod.assert_called_once()
        mock_core_api.create_namespaced_service.assert_called_once()

        # Verify resource naming
        rname = f"agent-{agent.id[:8]}"
        cm_call = mock_core_api.create_namespaced_config_map.call_args
        assert cm_call[0][0] == "test-ns"
        assert cm_call[0][1].metadata.name == rname

    @pytest.mark.asyncio
    async def test_spawn_env_forwarding(
        self,
        spawner: KubernetesSpawnerService,
        mock_core_api: MagicMock,
    ) -> None:
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
            patch.dict("os.environ", {"LLM_API_KEY": "test-key-123"}, clear=False),
        ):
            await spawner.spawn(
                SpawnRequest(
                    name="env-agent",
                    agent_type="stub",
                    env={"CUSTOM_VAR": "custom-val"},
                )
            )

        # Check env vars on the Pod container
        pod_call = mock_core_api.create_namespaced_pod.call_args
        pod_body = pod_call[0][1]
        env_vars = pod_body.spec.containers[0].env
        env_names = {e.name: e.value for e in env_vars}

        assert env_names["AGENT_ID"]
        assert env_names["AGENT_HTTP_PORT"] == "9000"
        assert env_names["LLM_API_KEY"] == "test-key-123"
        assert env_names["CUSTOM_VAR"] == "custom-val"

    @pytest.mark.asyncio
    async def test_kill_cleans_up_resources(
        self,
        spawner: KubernetesSpawnerService,
        mock_core_api: MagicMock,
    ) -> None:
        # Spawn first
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
        ):
            result = await spawner.spawn(
                SpawnRequest(name="kill-me", agent_type="stub")
            )

        agent = result.agent

        # Reset mocks to track kill-specific calls
        mock_core_api.reset_mock()

        await spawner.kill(agent.id)

        assert agent.state == AgentState.STOPPED
        mock_proc.terminate.assert_called()

        # Verify k8s resource deletion
        mock_core_api.delete_namespaced_service.assert_called_once()
        mock_core_api.delete_namespaced_pod.assert_called_once()
        mock_core_api.delete_namespaced_config_map.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_releases_team_capacity(
        self,
        spawner: KubernetesSpawnerService,
    ) -> None:
        await spawner.create_team(
            CreateTeamRequest(team_id="release-team", resources={"cpu": 1})
        )

        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
        ):
            result = await spawner.spawn(
                SpawnRequest(name="temp", team_id="release-team", agent_type="stub")
            )
            await spawner.kill(result.agent.id)

            # Should be able to spawn again now
            result2 = await spawner.spawn(
                SpawnRequest(name="temp2", team_id="release-team", agent_type="stub")
            )
            assert result2.agent.state == AgentState.RUNNING

    @pytest.mark.asyncio
    async def test_is_running_checks_pod_phase(
        self,
        spawner: KubernetesSpawnerService,
        mock_core_api: MagicMock,
    ) -> None:
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
        ):
            result = await spawner.spawn(
                SpawnRequest(name="check-running", agent_type="stub")
            )

        # Pod is Running
        assert await spawner.is_running(result.agent.id) is True

        # Pod goes to Failed
        mock_core_api.read_namespaced_pod.return_value = SimpleNamespace(
            status=SimpleNamespace(phase="Failed", conditions=[])
        )
        assert await spawner.is_running(result.agent.id) is False

    @pytest.mark.asyncio
    async def test_kill_nonexistent_agent_raises(
        self,
        spawner: KubernetesSpawnerService,
    ) -> None:
        with pytest.raises(KeyError):
            await spawner.kill("nonexistent-id")

    @pytest.mark.asyncio
    async def test_resource_name(self) -> None:
        assert (
            KubernetesSpawnerService._resource_name("abcdef12-3456") == "agent-abcdef12"
        )

    @pytest.mark.asyncio
    async def test_delete_team_kills_running_agents(
        self,
        spawner: KubernetesSpawnerService,
    ) -> None:
        await spawner.create_team(
            CreateTeamRequest(team_id="del-team", resources={"cpu": 2})
        )

        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
        ):
            result = await spawner.spawn(
                SpawnRequest(name="team-agent", team_id="del-team", agent_type="stub")
            )
            assert result.agent.state == AgentState.RUNNING

            await spawner.delete_team("del-team")
            assert result.agent.state == AgentState.STOPPED


# ── Resolver Tests ────────────────────────────────────────────────────


class TestKubernetesResolver:
    @pytest.mark.asyncio
    async def test_resolve_returns_localhost_url(
        self, agent_registry: AgentRegistry
    ) -> None:
        agent = Agent(
            id="res-agent-123",
            name="resolver-test",
            team_id="default",
            agent_type="openclaw",
            image_id="",
            http_port=12345,
        )
        await agent_registry.register(agent)

        resolver = KubernetesResolver(agent_registry)
        url = await resolver.resolve("res-agent-123")
        assert url == "http://127.0.0.1:12345"

    @pytest.mark.asyncio
    async def test_resolve_unknown_agent_raises(
        self, agent_registry: AgentRegistry
    ) -> None:
        resolver = KubernetesResolver(agent_registry)
        with pytest.raises(KeyError):
            await resolver.resolve("does-not-exist")


# ── Packaging Tests ───────────────────────────────────────────────────


class TestOciPackagingService:
    @pytest.mark.asyncio
    async def test_create_image_no_bundles(
        self, blob_store: LocalBlobStore, image_store: ImageStore
    ) -> None:
        svc = OciPackagingService(
            registry_url="registry.example.com",
            base_image="agentkernel:latest",
            downloader=URIDownloader(blob_store=blob_store),
            image_store=image_store,
        )
        job = await svc.create_agent_image(name="simple")
        assert job.status == "succeeded"
        assert job.image is not None
        assert job.image.name == "simple"
        assert "registry.example.com/simple:latest" == str(job.image.path)

        # Verify image is persisted in store
        loaded = image_store.get(job.image.id)
        assert loaded is not None
        assert str(loaded.path) == "registry.example.com/simple:latest"

    @pytest.mark.asyncio
    async def test_create_image_no_registry_url(
        self, blob_store: LocalBlobStore, image_store: ImageStore
    ) -> None:
        svc = OciPackagingService(
            registry_url="",
            base_image="agentkernel:latest",
            downloader=URIDownloader(blob_store=blob_store),
            image_store=image_store,
        )
        job = await svc.create_agent_image(name="local-img")
        assert job.status == "succeeded"
        assert "local-img:latest" == str(job.image.path)

    @pytest.mark.asyncio
    async def test_create_image_with_bundles(
        self, blob_store: LocalBlobStore, image_store: ImageStore, tmp_path: Path
    ) -> None:
        # Create a bundle
        helpers_dir = tmp_path / "helpers"
        helpers_dir.mkdir()
        (helpers_dir / "tools.py").write_text("def foo(): pass\n")
        bundle_uri = blob_store.upload_dir(helpers_dir)

        svc = OciPackagingService(
            registry_url="registry.example.com",
            base_image="agentkernel:latest",
            downloader=URIDownloader(blob_store=blob_store),
            image_store=image_store,
        )

        # Mock docker build + push
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            job = await svc.create_agent_image(
                name="bundled",
                bundles=[
                    SourceBundle(
                        uri=bundle_uri,
                        labels={"name": "myhelpers"},
                    )
                ],
            )

        assert job.status == "succeeded"
        assert job.image is not None
        assert "registry.example.com/bundled:" in str(job.image.path)

        # Verify image is persisted in store with registry tag
        loaded = image_store.get(job.image.id)
        assert loaded is not None
        assert "registry.example.com/bundled:" in str(loaded.path)

    @pytest.mark.asyncio
    async def test_create_image_podman_build_failure(
        self, blob_store: LocalBlobStore, image_store: ImageStore, tmp_path: Path
    ) -> None:
        helpers_dir = tmp_path / "helpers"
        helpers_dir.mkdir()
        (helpers_dir / "f.py").write_text("x = 1\n")
        bundle_uri = blob_store.upload_dir(helpers_dir)

        svc = OciPackagingService(
            registry_url="reg.io",
            base_image="base:latest",
            downloader=URIDownloader(blob_store=blob_store),
            image_store=image_store,
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"build error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            job = await svc.create_agent_image(
                name="fail",
                bundles=[SourceBundle(uri=bundle_uri, labels={"name": "x"})],
            )

        assert job.status == "failed"
        assert "podman build failed" in job.error

    @pytest.mark.asyncio
    async def test_create_image_with_requirements_txt(
        self, blob_store: LocalBlobStore, image_store: ImageStore, tmp_path: Path
    ) -> None:
        """Dockerfile should include pip install when bundles have requirements.txt."""
        helpers_dir = tmp_path / "helpers"
        helpers_dir.mkdir()
        (helpers_dir / "tools.py").write_text("def foo(): pass\n")
        (helpers_dir / "requirements.txt").write_text("requests>=2.28\n")
        bundle_uri = blob_store.upload_dir(helpers_dir)

        svc = OciPackagingService(
            registry_url="registry.example.com",
            base_image="agentkernel:latest",
            downloader=URIDownloader(blob_store=blob_store),
            image_store=image_store,
        )

        # Capture the Dockerfile contents via the build command
        dockerfiles: list[str] = []

        async def capture_dockerfile(*args, **kwargs):
            cwd = kwargs.get("cwd")
            if cwd:
                df_path = Path(cwd) / "Dockerfile"
                if df_path.exists():
                    dockerfiles.append(df_path.read_text())
            mock = AsyncMock()
            mock.returncode = 0
            mock.communicate = AsyncMock(return_value=(b"ok", b""))
            return mock

        with patch("asyncio.create_subprocess_exec", side_effect=capture_dockerfile):
            job = await svc.create_agent_image(
                name="with-reqs",
                bundles=[SourceBundle(uri=bundle_uri, labels={"name": "myhelpers"})],
            )

        assert job.status == "succeeded"
        assert len(dockerfiles) > 0
        assert "pip install" in dockerfiles[0]
        assert "requirements.txt" in dockerfiles[0]

    @pytest.mark.asyncio
    async def test_get_build_returns_none(
        self, blob_store: LocalBlobStore, image_store: ImageStore
    ) -> None:
        svc = OciPackagingService(
            registry_url="",
            base_image="base:latest",
            downloader=URIDownloader(blob_store=blob_store),
            image_store=image_store,
        )
        assert await svc.get_build("nonexistent") is None


# ── Spawner Image Tests ──────────────────────────────────────────────


class TestSpawnerImage:
    @pytest.mark.asyncio
    async def test_spawn_uses_request_image_when_set(
        self,
        spawner: KubernetesSpawnerService,
        image_store: ImageStore,
        mock_core_api: MagicMock,
    ) -> None:
        """When request.image_id is set, the spawner resolves it via ImageStore."""
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        # Pre-populate image store with a registry tag
        custom_tag = "registry.example.com/worker:abc123"
        image = image_store.create("img-uuid-123", "worker", registry_tag=custom_tag)

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
        ):
            await spawner.spawn(
                SpawnRequest(
                    name="custom-img", agent_type="stub", image_id="img-uuid-123"
                )
            )

        # Check the Pod container image
        pod_call = mock_core_api.create_namespaced_pod.call_args
        pod_body = pod_call[0][1]
        container_image = pod_body.spec.containers[0].image
        assert container_image == custom_tag

    @pytest.mark.asyncio
    async def test_spawn_with_unknown_image_id_raises(
        self,
        spawner: KubernetesSpawnerService,
    ) -> None:
        """When request.image_id is not found in ImageStore, spawner should error."""
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
            pytest.raises(KeyError, match="Image not found"),
        ):
            await spawner.spawn(
                SpawnRequest(
                    name="bad-img", agent_type="stub", image_id="nonexistent-uuid"
                )
            )

    @pytest.mark.asyncio
    async def test_spawn_uses_base_image_when_request_image_is_none(
        self,
        spawner: KubernetesSpawnerService,
        mock_core_api: MagicMock,
    ) -> None:
        """When request.image_id is None, the spawner should use base_image."""
        mock_proc = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(spawner, "_wait_for_local_ready", new_callable=AsyncMock),
        ):
            await spawner.spawn(SpawnRequest(name="default-img", agent_type="stub"))

        # Check the Pod container image
        pod_call = mock_core_api.create_namespaced_pod.call_args
        pod_body = pod_call[0][1]
        container_image = pod_body.spec.containers[0].image
        assert container_image == "registry.example.com/agentkernel:latest"
