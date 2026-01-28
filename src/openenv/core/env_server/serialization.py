# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared serialization and deserialization utilities for OpenEnv HTTP servers.

This module provides common utilities for converting between JSON dictionaries
and Pydantic models (Action/Observation) to eliminate code duplication across
HTTP server and web interface implementations.
"""

from typing import Any, Dict, Type, get_origin, get_args, Union

from pydantic import TypeAdapter

from .types import Action, Observation


# Cache for TypeAdapters to avoid recreation
_type_adapter_cache: Dict[Type, TypeAdapter] = {}


def _is_mcp_action_type(action_cls: Type[Action]) -> bool:
    """Check if action class is an MCP action type that needs special handling."""
    # Import here to avoid circular imports
    from .mcp_types import ListToolsAction, CallToolAction

    # Check if it's one of the MCP action types or a Union containing them
    if action_cls in (ListToolsAction, CallToolAction):
        return True

    # Check for Union types (including Annotated unions)
    origin = get_origin(action_cls)
    if origin is Union:
        args = get_args(action_cls)
        return any(
            arg in (ListToolsAction, CallToolAction)
            or getattr(arg, "__origin__", None) is type
            and issubclass(arg, (ListToolsAction, CallToolAction))
            for arg in args
        )

    return False


def _get_type_adapter(action_cls: Type[Action]) -> TypeAdapter:
    """Get or create a TypeAdapter for the action class."""
    if action_cls not in _type_adapter_cache:
        _type_adapter_cache[action_cls] = TypeAdapter(action_cls)
    return _type_adapter_cache[action_cls]


def deserialize_action(action_data: Dict[str, Any], action_cls: Type[Action]) -> Action:
    """
    Convert JSON dict to Action instance using Pydantic validation.

    This is a basic deserialization that works for most environments.
    For special cases (e.g., tensor fields, custom type conversions),
    use deserialize_action_with_preprocessing().

    For MCP actions (MCPAction union type), this uses a TypeAdapter to
    handle discriminated union deserialization based on the 'type' field.

    Args:
        action_data: Dictionary containing action data
        action_cls: The Action subclass to instantiate (can be a Union type)

    Returns:
        Action instance

    Raises:
        ValidationError: If action_data is invalid for the action class

    Note:
        This uses Pydantic's model_validate() for automatic validation.
        For Union types (like MCPAction), it uses TypeAdapter.validate_python().
    """
    # Import MCP types to check for MCPAction
    from .mcp_types import MCPAction, ListToolsAction, CallToolAction

    # Check if we need to use MCPAction for polymorphic deserialization
    # This handles both when action_cls is MCPAction itself or CallToolAction
    # (since CallToolAction is often used as the "default" MCP action type)
    if action_cls is CallToolAction:
        # Check if the action data has a different type field
        action_type = action_data.get("type", "call_tool")
        if action_type == "list_tools":
            # Use MCPAction union for proper deserialization
            adapter = _get_type_adapter(MCPAction)
            return adapter.validate_python(action_data)

    # Check if this is already a Union type (like MCPAction)
    origin = get_origin(action_cls)
    if origin is Union or hasattr(action_cls, "__origin__"):
        adapter = _get_type_adapter(action_cls)
        return adapter.validate_python(action_data)

    # Standard model_validate for simple action types
    return action_cls.model_validate(action_data)


def deserialize_action_with_preprocessing(
    action_data: Dict[str, Any], action_cls: Type[Action]
) -> Action:
    """
    Convert JSON dict to Action instance with preprocessing for special types.

    This version handles common type conversions needed for web interfaces:
    - Converting lists/strings to tensors for 'tokens' field
    - Converting string action_id to int
    - Other custom preprocessing as needed

    Args:
        action_data: Dictionary containing action data
        action_cls: The Action subclass to instantiate

    Returns:
        Action instance

    Raises:
        ValidationError: If action_data is invalid for the action class
    """
    processed_data = {}

    for key, value in action_data.items():
        if key == "tokens" and isinstance(value, (list, str)):
            # Convert list or string to tensor
            if isinstance(value, str):
                # If it's a string, try to parse it as a list of numbers
                try:
                    import json

                    value = json.loads(value)
                except Exception:
                    # If parsing fails, treat as empty list
                    value = []
            if isinstance(value, list):
                try:
                    import torch  # type: ignore

                    processed_data[key] = torch.tensor(value, dtype=torch.long)
                except ImportError:
                    # If torch not available, keep as list
                    processed_data[key] = value
            else:
                processed_data[key] = value
        elif key == "action_id" and isinstance(value, str):
            # Convert action_id from string to int
            try:
                processed_data[key] = int(value)
            except ValueError:
                # If conversion fails, keep original value
                processed_data[key] = value
        else:
            processed_data[key] = value

    return action_cls.model_validate(processed_data)


def serialize_observation(observation: Observation) -> Dict[str, Any]:
    """
    Convert Observation instance to JSON-compatible dict using Pydantic.

    Args:
        observation: Observation instance

    Returns:
        Dictionary compatible with EnvClient._parse_result()

    The format matches what EnvClient expects:
    {
        "observation": {...},  # Observation fields
        "reward": float | None,
        "done": bool,
    }
    """
    # Use Pydantic's model_dump() for serialization
    obs_dict = observation.model_dump(
        exclude={
            "reward",
            "done",
            "metadata",
        }  # Exclude these from observation dict
    )

    # Extract reward and done directly from the observation
    reward = observation.reward
    done = observation.done

    # Return in EnvClient expected format
    return {
        "observation": obs_dict,
        "reward": reward,
        "done": done,
    }
