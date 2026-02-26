# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Protocol definitions for backend-aware kernel services."""

from typing import Protocol

from ..config import Agent, CreateTeamRequest, PackageJob, SourceBundle, SpawnRequest


class SpawnerService(Protocol):
    """Interface for agent lifecycle management.

    Spawner services handle everything: team provisioning, agent
    process lifecycle, image resolution, workspace creation, port
    allocation, and networking.

    Implementations:
    - LocalSpawnerService: subprocesses, no isolation
    - BwrapSpawnerService: bubblewrap sandboxing
    - KubernetesSpawnerService: pods in a k8s cluster
    """

    async def create_team(self, request: CreateTeamRequest) -> None:
        """Reserve capacity for a team.

        The request.config field is an opaque payload interpreted by
        each implementation:
        - Local/Bwrap: ignored
        - Kubernetes: namespace, kubeconfig, node_selector, ...
        - Pulumi: provider, region, instance_type, ...
        """
        ...

    async def delete_team(self, team_id: str) -> None:
        """Delete a team and release its capacity.

        Kills all running agents in the team first, then tears down
        any provisioned infrastructure.
        """
        ...

    async def spawn(self, request: SpawnRequest) -> Agent:
        """Spawn an agent from a SpawnRequest.

        Creates the agent record, resolves images, allocates ports,
        creates workspace, and starts the process.

        If request.team_id refers to a team created via create_team(),
        capacity is enforced. Otherwise, no capacity enforcement.
        """
        ...

    async def kill(self, agent_id: str) -> None:
        """Kill an agent and release its resources."""
        ...

    async def get(self, agent_id: str) -> Agent | None:
        """Look up an agent by ID."""
        ...

    async def is_running(self, agent_id: str) -> bool:
        """Check if an agent's process is still alive."""
        ...


class Resolver(Protocol):
    """Maps an agent_id to its HTTP base URL.

    This is the only backend-specific piece of agent communication.
    The AgentClient (in agent_client.py) handles the shared HTTP+SSE
    transport; the Resolver handles routing.

    Implementations:
    - LocalResolver: returns http://127.0.0.1:{http_port}
    - KubernetesResolver: would return k8s Service DNS URL
    """

    async def resolve(self, agent_id: str) -> str:
        """Return the base URL for the given agent (e.g. "http://127.0.0.1:9000")."""
        ...


class PackagingService(Protocol):
    """Interface for packaging agent images.

    Packaging services turn source bundles into deployable agent images.
    The mechanism is backend-specific:

    Implementations:
    - LocalPackagingService: copies bundles into a local directory
    - OciPackagingService: builds an OCI image and pushes to a registry
    """

    async def create_agent_image(
        self, name: str, bundles: list[SourceBundle] | None = None
    ) -> PackageJob:
        """Package an agent image from bundles."""
        ...

    async def get_build(self, build_id: str) -> PackageJob | None:
        """Look up a packaging job by ID."""
        ...
