# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""AgentTypePlugin protocol — interface for pluggable agent types.

Each agent type (e.g. openclaw, ...) implements this protocol
to provide its spawn-time config builder and launch command. The spawner
dispatches generically via a plugin registry instead of hardcoded
if/elif/else chains.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config import Agent, SpawnRequest


@runtime_checkable
class AgentTypePlugin(Protocol):
    """Protocol that agent-type plugins must satisfy.

    Split into build_config + resolve_command for flexibility — spawners
    like bwrap/k8s may need to handle the command separately (e.g.,
    wrapping in bwrap, putting in a pod spec) while keeping the config
    as-is.
    """

    @property
    def agent_type(self) -> str:
        """Unique identifier, e.g. 'openclaw'."""
        ...

    def build_config(
        self,
        agent: Agent,
        nonce: str,
        http_port: int,
        workspace_path: Path,
        request: SpawnRequest,
    ) -> dict[str, Any]:
        """Build agent-type-specific config dict.

        The spawner appends cross-cutting agentbus fields after this call.
        Should validate spawn_info type and raise TypeError if wrong.
        """
        ...

    def resolve_command(self) -> list[str]:
        """Return the launch command for this agent type.

        Python agents: [sys.executable, "-m", "some.module"]
        Binary agents: ["/path/to/binary"]
        Raises FileNotFoundError if required binary not found.
        """
        ...
