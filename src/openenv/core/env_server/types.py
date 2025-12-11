# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# Type aliases
Scalar = Union[int, float, bool]


class Action(BaseModel):
    """Base class for all environment actions.

    All action subclasses should inherit from this base class.
    Uses Pydantic for automatic validation and serialization.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True,  # Validate on field assignment
        arbitrary_types_allowed=True,  # Allow numpy arrays, torch tensors, etc.
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the action"
    )


class Observation(BaseModel):
    """Base class for all environment observations.

    All observation subclasses should inherit from this base class.
    Uses Pydantic for automatic validation and serialization.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: bool | int | float | None = Field(
        default=None, description="Reward signal from the last action"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the observation"
    )


class ResetRequest(BaseModel):
    """Request model for environment reset."""

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for custom reset parameters
        json_schema_extra={"examples": [{"seed": 42, "episode_id": "episode-001"}, {}]},
    )

    seed: Optional[int] = Field(
        default=None, ge=0, description="Random seed for reproducible episodes"
    )
    episode_id: Optional[str] = Field(
        default=None, max_length=255, description="Custom episode identifier"
    )


class ResetResponse(BaseModel):
    """Response model for environment reset."""

    model_config = ConfigDict(extra="forbid")

    observation: Dict[str, Any] = Field(
        ..., description="Initial observation from the environment"
    )
    reward: Optional[float] = Field(
        default=None, description="Initial reward (typically None at reset)"
    )
    done: bool = Field(
        default=False, description="Whether episode is already done (typically False)"
    )


class StepRequest(BaseModel):
    """Request model for environment step."""

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for custom step parameters
        json_schema_extra={
            "examples": [
                {"action": {"value": 1}, "timeout_s": 30.0},
                {"action": {"value": 1}, "render": True, "verbose": False},
            ]
        },
    )

    action: Dict[str, Any] = Field(
        ...,
        description="Action to execute, must conform to environment's action schema",
    )
    timeout_s: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional timeout in seconds for action execution",
    )
    request_id: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional request identifier for tracking",
    )


class StepResponse(BaseModel):
    """Response model for environment step."""

    model_config = ConfigDict(extra="forbid")

    observation: Dict[str, Any] = Field(
        ..., description="Observation resulting from the action"
    )
    reward: Optional[float] = Field(
        default=None, description="Reward signal from the action"
    )
    done: bool = Field(default=False, description="Whether the episode has terminated")


class State(BaseModel):
    """Base class for environment state.

    Represents internal environment state, separate from observations.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    episode_id: Optional[str] = Field(
        default=None, description="Unique identifier for the current episode"
    )
    step_count: int = Field(
        default=0,
        ge=0,  # Greater than or equal to 0
        description="Number of steps taken in the current episode",
    )


class CodeExecResult(BaseModel):
    """Result of code execution containing stdout, stderr, and exit code."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    stdout: str = Field(description="Standard output from code execution")
    stderr: str = Field(description="Standard error from code execution")
    exit_code: int = Field(description="Exit code from code execution")


class EnvironmentMetadata(BaseModel):
    """Metadata about an environment for documentation and UI purposes."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(description="Name of the environment")
    description: str = Field(description="Description of what the environment does")
    readme_content: Optional[str] = Field(
        default=None, description="Content of the README file for the environment"
    )
    version: Optional[str] = Field(
        default=None, description="Version of the environment"
    )
    author: Optional[str] = Field(default=None, description="Author of the environment")
    documentation_url: Optional[str] = Field(
        default=None, description="URL to the environment's documentation"
    )


class SchemaResponse(BaseModel):
    """Response model for the combined schema endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    action: Dict[str, Any] = Field(
        description="JSON schema for actions accepted by this environment"
    )
    observation: Dict[str, Any] = Field(
        description="JSON schema for observations returned by this environment"
    )
    state: Dict[str, Any] = Field(
        description="JSON schema for environment state objects"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    status: str = Field(description="Health status of the environment server")

class WSMessage(BaseModel):
    """Base class for WebSocket messages."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    type: str = Field(description="Message type identifier")


class WSResetMessage(WSMessage):
    """WebSocket message to reset the environment."""

    type: str = Field(default="reset", description="Message type")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional reset parameters (seed, episode_id, etc.)",
    )


class WSStepMessage(WSMessage):
    """WebSocket message to execute a step."""

    type: str = Field(default="step", description="Message type")
    data: Dict[str, Any] = Field(
        ..., description="Action data conforming to environment's action schema"
    )


class WSStateMessage(WSMessage):
    """WebSocket message to request current state."""

    type: str = Field(default="state", description="Message type")


class WSCloseMessage(WSMessage):
    """WebSocket message to close the session."""

    type: str = Field(default="close", description="Message type")


class WSObservationResponse(BaseModel):
    """WebSocket response containing an observation."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="observation", description="Response type")
    data: Dict[str, Any] = Field(description="Observation data")


class WSStateResponse(BaseModel):
    """WebSocket response containing environment state."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="state", description="Response type")
    data: Dict[str, Any] = Field(description="State data")


class WSErrorResponse(BaseModel):
    """WebSocket response for errors."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="error", description="Response type")
    data: Dict[str, Any] = Field(description="Error details including message and code")


class ConcurrencySafetyLevel(str):
    """
    Classification of environment concurrency safety.
    
    Environments are classified based on their ability to safely handle
    multiple concurrent sessions within a single container.
    """
    
    UNSAFE = "unsafe"
    SAFE = "safe"


class ConcurrencyConfig(BaseModel):
    """Configuration for concurrent environment sessions."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    max_concurrent_envs: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Maximum number of concurrent WebSocket sessions allowed",
    )
    session_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout in seconds for inactive sessions. None means no timeout.",
    )
    reject_on_capacity: bool = Field(
        default=True,
        description="If True, reject new connections when at capacity. If False, queue them.",
    )


class ServerCapacityStatus(BaseModel):
    """Status of server capacity for concurrent sessions."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    active_sessions: int = Field(
        ge=0,
        description="Number of currently active sessions",
    )
    max_sessions: int = Field(
        ge=1,
        description="Maximum number of allowed sessions",
    )
    available_slots: int = Field(
        ge=0,
        description="Number of available session slots",
    )
    is_at_capacity: bool = Field(
        description="Whether the server has reached maximum capacity",
    )

    @classmethod
    def from_counts(cls, active: int, max_sessions: int) -> "ServerCapacityStatus":
        """Create status from active and max session counts."""
        available = max(0, max_sessions - active)
        return cls(
            active_sessions=active,
            max_sessions=max_sessions,
            available_slots=available,
            is_at_capacity=active >= max_sessions,
        )


class SessionInfo(BaseModel):
    """Information about an active session."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    session_id: str = Field(description="Unique identifier for the session")
    created_at: float = Field(description="Unix timestamp when the session was created")
    last_activity_at: float = Field(
        description="Unix timestamp of the last activity in the session"
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps executed in this session",
    )
    environment_type: str = Field(
        description="Type name of the environment class for this session"
    )
