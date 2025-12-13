# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for FleetEnvClient (RFC 003 tool-call actions)."""

from dataclasses import dataclass, field
from typing import Any, Dict, TYPE_CHECKING

# Avoid importing OpenAI typing aliases at runtime.
# The `openai` package changes exported type names across major versions, and
# Fleet integration should work even if OpenAI isn't installed.
if TYPE_CHECKING:  # pragma: no cover
    try:
        from openai import ChatCompletionToolUnionParam as OpenAIToolParam  # type: ignore
    except Exception:  # noqa: BLE001
        OpenAIToolParam = Dict[str, Any]  # type: ignore[misc,assignment]
else:
    OpenAIToolParam = Dict[str, Any]  # type: ignore[misc,assignment]

from mcp.types import Tool


# Support both in-repo and standalone imports
try:
    from core.env_server.types import Action
except ImportError:
    from openenv_core.env_server.types import Action

def normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return schema

    result = {}

    if "anyOf" in schema:
        non_null_schemas = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if non_null_schemas:
            schema = {**schema, **non_null_schemas[0]}
            del schema["anyOf"]

    for key, value in schema.items():
        if key in ["title", "default", "anyOf"]:
            continue

        if key == "prefixItems":
            result["items"] = (
                normalize_schema(value[0]) if value else {"type": "string"}
            )
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {k: normalize_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = normalize_schema(value)
        else:
            result[key] = value

    return result


def convert_tool_format(tool: Tool) -> OpenAIToolParam:
    normalized_properties = {
        key: normalize_schema(value)
        for key, value in tool.inputSchema.get("properties", {}).items()
    }

    # OpenAI "tools" format: {"type": "function", "function": {...}}
    openai_tool: OpenAIToolParam = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": normalized_properties,
                "required": tool.inputSchema.get("required", []),
            },
        },
    }
    return openai_tool


@dataclass(kw_only=True)
class ListToolsAction(Action):
    """Request list of available MCP tools from the Fleet environment."""

    tools: list[OpenAIToolParam] = field(default_factory=list)


@dataclass(kw_only=True)
class CallToolAction(Action):
    """Call a specific MCP tool exposed by the Fleet environment."""

    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


