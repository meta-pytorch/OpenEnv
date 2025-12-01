# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for MCP types: Tool, ToolError, ToolErrorType, and related observations.

These tests validate the strongly typed MCP types work correctly.
"""

import pytest
from dataclasses import asdict

from core.env_server.mcp_types import (
    Tool,
    ToolError,
    ToolErrorType,
    ListToolsObservation,
    CallToolObservation,
    ListToolsAction,
    CallToolAction,
)


class TestTool:
    """Tests for the Tool dataclass."""

    def test_tool_creation_basic(self):
        """Test basic Tool creation with required fields."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object", "properties": {"arg1": {"type": "string"}}}
        assert tool.output_schema is None

    def test_tool_creation_with_output_schema(self):
        """Test Tool creation with optional output_schema."""
        tool = Tool(
            name="echo",
            description="Echo a message",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"echoed": {"type": "string"}}},
        )
        assert tool.name == "echo"
        assert tool.output_schema == {"type": "object", "properties": {"echoed": {"type": "string"}}}

    def test_tool_serialization(self):
        """Test that Tool can be serialized via asdict."""
        tool = Tool(
            name="add",
            description="Add two numbers",
            input_schema={"type": "object"},
        )
        serialized = asdict(tool)
        assert serialized == {
            "name": "add",
            "description": "Add two numbers",
            "input_schema": {"type": "object"},
            "output_schema": None,
        }

    def test_tool_equality(self):
        """Test Tool equality comparison."""
        tool1 = Tool(name="tool", description="desc", input_schema={})
        tool2 = Tool(name="tool", description="desc", input_schema={})
        tool3 = Tool(name="other", description="desc", input_schema={})
        
        assert tool1 == tool2
        assert tool1 != tool3


class TestToolError:
    """Tests for the ToolError dataclass."""

    def test_tool_error_creation(self):
        """Test basic ToolError creation."""
        error = ToolError(
            error_type=ToolErrorType.EXECUTION_ERROR,
            message="Something went wrong",
        )
        assert error.error_type == ToolErrorType.EXECUTION_ERROR
        assert error.message == "Something went wrong"
        assert error.details is None

    def test_tool_error_with_details(self):
        """Test ToolError with additional details."""
        error = ToolError(
            error_type=ToolErrorType.INVALID_ARGUMENTS,
            message="Missing required argument",
            details={"missing_field": "arg1", "provided_fields": ["arg2"]},
        )
        assert error.error_type == ToolErrorType.INVALID_ARGUMENTS
        assert error.details == {"missing_field": "arg1", "provided_fields": ["arg2"]}

    def test_tool_error_serialization(self):
        """Test that ToolError can be serialized via asdict."""
        error = ToolError(
            error_type=ToolErrorType.TIMEOUT,
            message="Request timed out",
        )
        serialized = asdict(error)
        assert serialized == {
            "error_type": ToolErrorType.TIMEOUT,
            "message": "Request timed out",
            "details": None,
        }


class TestToolErrorType:
    """Tests for the ToolErrorType enum."""

    def test_all_error_types_exist(self):
        """Test all expected error types are defined."""
        expected_types = [
            "INVALID_ARGUMENTS",
            "TOOL_NOT_FOUND",
            "TRANSPORT_ERROR",
            "EXECUTION_ERROR",
            "TIMEOUT",
            "INTERNAL_ERROR",
        ]
        for error_type in expected_types:
            assert hasattr(ToolErrorType, error_type)

    def test_error_type_values(self):
        """Test error type string values."""
        assert ToolErrorType.INVALID_ARGUMENTS.value == "invalid_arguments"
        assert ToolErrorType.TOOL_NOT_FOUND.value == "tool_not_found"
        assert ToolErrorType.TRANSPORT_ERROR.value == "transport_error"
        assert ToolErrorType.EXECUTION_ERROR.value == "execution_error"
        assert ToolErrorType.TIMEOUT.value == "timeout"
        assert ToolErrorType.INTERNAL_ERROR.value == "internal_error"

    def test_error_type_from_value(self):
        """Test creating ToolErrorType from string value."""
        assert ToolErrorType("execution_error") == ToolErrorType.EXECUTION_ERROR
        assert ToolErrorType("internal_error") == ToolErrorType.INTERNAL_ERROR


class TestListToolsObservation:
    """Tests for the ListToolsObservation dataclass."""

    def test_list_tools_observation_empty(self):
        """Test ListToolsObservation with empty tools list."""
        obs = ListToolsObservation()
        assert obs.done is False
        assert obs.tools == []

    def test_list_tools_observation_with_tools(self):
        """Test ListToolsObservation with typed Tool objects."""
        tools = [
            Tool(name="tool1", description="First tool", input_schema={}),
            Tool(name="tool2", description="Second tool", input_schema={"type": "object"}),
        ]
        obs = ListToolsObservation(tools=tools)
        
        assert len(obs.tools) == 2
        assert obs.tools[0].name == "tool1"
        assert obs.tools[1].name == "tool2"
        assert isinstance(obs.tools[0], Tool)
        assert isinstance(obs.tools[1], Tool)

    def test_list_tools_observation_serialization(self):
        """Test ListToolsObservation serialization."""
        tool = Tool(name="echo", description="Echo tool", input_schema={})
        obs = ListToolsObservation(tools=[tool])
        serialized = asdict(obs)
        
        assert "tools" in serialized
        assert len(serialized["tools"]) == 1
        assert serialized["tools"][0]["name"] == "echo"


class TestCallToolObservation:
    """Tests for the CallToolObservation dataclass."""

    def test_call_tool_observation_success(self):
        """Test CallToolObservation for a successful tool call."""
        obs = CallToolObservation(
            tool_name="echo",
            result={"message": "Hello!"},
        )
        assert obs.tool_name == "echo"
        assert obs.result == {"message": "Hello!"}
        assert obs.error is None
        assert obs.done is False

    def test_call_tool_observation_with_error(self):
        """Test CallToolObservation with a ToolError."""
        error = ToolError(
            error_type=ToolErrorType.TOOL_NOT_FOUND,
            message="Tool 'missing' not found",
        )
        obs = CallToolObservation(
            tool_name="missing",
            result=None,
            error=error,
        )
        assert obs.tool_name == "missing"
        assert obs.result is None
        assert obs.error is not None
        assert obs.error.error_type == ToolErrorType.TOOL_NOT_FOUND

    def test_call_tool_observation_serialization(self):
        """Test CallToolObservation serialization."""
        obs = CallToolObservation(
            tool_name="add",
            result={"sum": 10},
        )
        serialized = asdict(obs)
        
        assert serialized["tool_name"] == "add"
        assert serialized["result"] == {"sum": 10}
        assert serialized["error"] is None


class TestActions:
    """Tests for MCP action types."""

    def test_list_tools_action(self):
        """Test ListToolsAction creation."""
        action = ListToolsAction()
        assert action.metadata == {}

    def test_call_tool_action(self):
        """Test CallToolAction creation."""
        action = CallToolAction(
            tool_name="echo",
            parameters={"message": "Hello"},
        )
        assert action.tool_name == "echo"
        assert action.parameters == {"message": "Hello"}

    def test_call_tool_action_empty_parameters(self):
        """Test CallToolAction with default empty parameters."""
        action = CallToolAction(tool_name="no_args_tool")
        assert action.tool_name == "no_args_tool"
        assert action.parameters == {}


@pytest.mark.asyncio
async def test_mcp_environment_returns_typed_tools():
    """Test that MCPEnvironment returns properly typed Tool objects."""
    from envs.echo_env.server.echo_environment import EchoEnvironment

    env = EchoEnvironment()
    
    # Get tools via ListToolsAction
    list_action = ListToolsAction()
    obs = await env._handle_mcp_action(list_action)
    
    # Verify we get Tool objects, not dicts
    assert len(obs.tools) > 0
    for tool in obs.tools:
        assert isinstance(tool, Tool)
        assert isinstance(tool.name, str)
        assert isinstance(tool.description, str)
        assert isinstance(tool.input_schema, dict)


@pytest.mark.asyncio
async def test_mcp_environment_error_handling():
    """Test that MCPEnvironment properly handles errors with ToolError."""
    from envs.echo_env.server.echo_environment import EchoEnvironment

    env = EchoEnvironment()
    
    # Try calling a non-existent tool
    action = CallToolAction(
        tool_name="nonexistent_tool",
        parameters={},
    )
    obs = await env._handle_mcp_action(action)
    
    # Should have an error with ToolError type
    assert obs.error is not None
    assert isinstance(obs.error, ToolError)
    assert isinstance(obs.error.error_type, ToolErrorType)
    assert obs.error.message  # Should have an error message
