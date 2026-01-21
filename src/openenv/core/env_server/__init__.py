# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core environment interfaces and types."""

from .base_transforms import CompositeTransform, NullTransform
from .http_server import HTTPEnvServer, create_app, create_fastapi_app
from .interfaces import Environment, Message, ModelTokenizer, Transform
from .route_config import GetEndpointConfig
from .serialization import (
    deserialize_action,
    deserialize_action_with_preprocessing,
    serialize_observation,
)
from .types import (
    Action,
    Observation,
    State,
    SchemaResponse,
    HealthResponse,
    BaseMessage,
    WSIncomingMessage,
    WSResetMessage,
    WSStepMessage,
    WSStateMessage,
    WSCloseMessage,
    WSObservationResponse,
    WSStateResponse,
    WSErrorResponse,
    ConcurrencyConfig,
    ServerCapacityStatus,
    SessionInfo,
)
from .exceptions import (
    OpenEnvError,
    ConcurrencyConfigurationError,
    SessionCapacityError,
    SessionNotFoundError,
    SessionCreationError,
    EnvironmentFactoryError,
)
from .web_interface import create_web_interface_app, WebInterfaceManager
from .mcp_types import (
    Tool,
    ToolError,
    ToolErrorType,
    ListToolsAction,
    CallToolAction,
    ListToolsObservation,
    CallToolObservation,
    WSMCPMessage,
    WSMCPResponse,
    RESERVED_TOOL_NAMES,
)
from .mcp_environment import MCPEnvironment

__all__ = [
    # Core interfaces
    "Environment",
    "Transform",
    "Message",
    "ModelTokenizer",
    # Types
    "Action",
    "Observation",
    "State",
    "SchemaResponse",
    "HealthResponse",
    # WebSocket message types
    "BaseMessage",
    "WSIncomingMessage",
    "WSResetMessage",
    "WSStepMessage",
    "WSStateMessage",
    "WSCloseMessage",
    "WSObservationResponse",
    "WSStateResponse",
    "WSErrorResponse",
    # Concurrency types
    "ConcurrencyConfig",
    "ServerCapacityStatus",
    "SessionInfo",
    # Exceptions
    "OpenEnvError",
    "ConcurrencyConfigurationError",
    "SessionCapacityError",
    "SessionNotFoundError",
    "SessionCreationError",
    "EnvironmentFactoryError",
    # Base transforms
    "CompositeTransform",
    "NullTransform",
    # HTTP Server
    "HTTPEnvServer",
    "create_app",
    "create_fastapi_app",
    # Web Interface
    "create_web_interface_app",
    "WebInterfaceManager",
    # Serialization utilities
    "deserialize_action",
    "deserialize_action_with_preprocessing",
    "serialize_observation",
    # Route configuration
    "GetEndpointConfig",
    # MCP types
    "Tool",
    "ToolError",
    "ToolErrorType",
    "ListToolsAction",
    "CallToolAction",
    "ListToolsObservation",
    "CallToolObservation",
    "WSMCPMessage",
    "WSMCPResponse",
    "RESERVED_TOOL_NAMES",
    "MCPEnvironment",
]
