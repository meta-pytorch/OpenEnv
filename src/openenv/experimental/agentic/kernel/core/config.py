# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Core data types for the agentkernel system."""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass
class AgentBusConfig:
    """Kernel-level AgentBus configuration.

    url: AgentBus connection URL. Supported schemes:
        memory://        - In-process server, random port (kernel will allocate)
        memory://<port>  - In-process server, fixed port
        remote://h:p     - Connect to existing gRPC server
    disable_safety: Skip safety voting/decisions.
    """

    url: str = "memory://"
    disable_safety: bool = False


class AgentState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    STOPPED = "stopped"
    FAILED = "failed"


class TransportFormat(Enum):
    ANTHROPIC = "anthropic"
    OAT = "oat"


@dataclass
class AgentImage:
    id: str
    name: str
    path: Path


@dataclass
class Agent:
    """A running agent in the kernel.

    Agents are private to their spawner — one agent, one conversation,
    one owner. Only the entity that spawned an agent receives the nonce
    (via SpawnResult) needed to issue Turn requests. Without this,
    any caller could talk to any agent, which would require a Session
    or Conversation container for independent state — but Mahesh and
    David strongly feel that sessions add complexity to reasoning about
    state changes and how one conversation affects another. So we have
    no sessions; the nonce enforces single-owner access instead.

    The Agent record itself is visible in team listings but does not
    carry the nonce.

    Generic fields (name, team_id, agent_type, etc.) are shared across
    all agent types. Cross-cutting labels like ``role`` belong in
    ``metadata`` (e.g. ``metadata={"role": "coordinator"}``).
    Agent-type-specific config lives in spawn_info — a typed dataclass
    defined by the agent type's plugin (e.g. plugins.openclaw.OpenClawSpawnInfo).
    """

    id: str
    name: str
    team_id: str
    agent_type: str
    image_id: str
    spawn_info: Any = (
        None  # Agent-type-specific: e.g. plugins.openclaw.OpenClawSpawnInfo
    )
    metadata: dict[str, str] = field(default_factory=dict)
    state: AgentState = AgentState.STARTING
    pid: int | None = None
    http_port: int | None = None
    agentbus: AgentBusConfig | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class SpawnResult:
    """Returned by spawner.spawn().

    Contains the Agent record plus a nonce that the spawner must present
    in every TurnRequest. The nonce is a secret — it is NOT stored on
    the Agent and is NOT visible in team listings or other read APIs.
    """

    agent: Agent
    nonce: str


@dataclass
class SourceBundle:
    uri: str
    labels: dict[str, str]


@dataclass
class PackageJob:
    id: str
    status: str
    error: str | None = None
    image: AgentImage | None = None


@dataclass
class CreateTeamRequest:
    """Request to create a team with reserved capacity.

    The config field is an opaque payload interpreted by the spawner
    service implementation:
    - Local/Bwrap: ignored
    - Kubernetes: namespace, kubeconfig, node_selector, ...
    - Pulumi: provider, region, instance_type, ...
    """

    team_id: str
    resources: dict[str, int] = field(default_factory=lambda: {"cpu": 4})
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpawnRequest:
    """Request to spawn an agent.

    Generic fields (name, agent_type, etc.) apply to all agent types.
    Cross-cutting labels like ``role`` belong in ``metadata``
    (e.g. ``metadata={"role": "worker"}``). Agent-type-specific config
    goes in spawn_info — a typed dataclass defined by the agent type's
    plugin (e.g. plugins/openclaw.py OpenClawSpawnInfo).
    """

    name: str
    team_id: str = ""
    agent_type: str = ""
    image_id: str | None = None
    transport_format: TransportFormat = TransportFormat.ANTHROPIC
    spawn_info: Any = (
        None  # Agent-type-specific: e.g. plugins.openclaw.OpenClawSpawnInfo
    )
    agentbus: AgentBusConfig | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class TurnRequest:
    agent_id: str
    nonce: str = ""
    body: bytes = b""


@dataclass
class TurnResponse:
    body: str
    done: bool
    error: str | None = None
