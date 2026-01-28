# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MCP type definitions."""

import pytest
from pydantic import ValidationError

from openenv.core.env_server.mcp_types import (
    Tool,
    ToolError,
    ToolErrorType,
    ListToolsAction,
    CallToolAction,
    ListToolsObservation,
    CallToolObservation,
    WSMCPMessage,
    WSMCPResponse,
    MCPRequest,
    MCPResponse,
    RESERVED_TOOL_NAMES,
)


class TestTool:
    """Tests for the Tool model."""

    def test_tool_creation(self):
        """Test creating a valid Tool."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "properties" in tool.input_schema

    def test_tool_requires_all_fields(self):
        """Test that Tool requires name, description, and input_schema."""
        with pytest.raises(ValidationError):
            Tool(name="test")  # Missing description and input_schema

    def test_tool_serialization(self):
        """Test Tool can be serialized to dict."""
        tool = Tool(
            name="echo",
            description="Echo message",
            input_schema={"type": "object"},
        )
        data = tool.model_dump()
        assert data["name"] == "echo"
        assert data["description"] == "Echo message"


class TestToolError:
    """Tests for the ToolError model."""

    def test_tool_error_creation(self):
        """Test creating a ToolError."""
        error = ToolError(
            error_type=ToolErrorType.EXECUTION_ERROR,
            message="Something went wrong",
        )
        assert error.error_type == ToolErrorType.EXECUTION_ERROR
        assert error.message == "Something went wrong"

    def test_all_error_types(self):
        """Test all error types can be used."""
        for error_type in ToolErrorType:
            error = ToolError(error_type=error_type, message="test")
            assert error.error_type == error_type


class TestListToolsAction:
    """Tests for ListToolsAction."""

    def test_list_tools_action_creation(self):
        """Test creating a ListToolsAction."""
        action = ListToolsAction()
        assert action.type == "list_tools"

    def test_list_tools_action_metadata(self):
        """Test ListToolsAction supports metadata."""
        action = ListToolsAction(metadata={"request_id": "123"})
        assert action.metadata["request_id"] == "123"


class TestCallToolAction:
    """Tests for CallToolAction."""

    def test_call_tool_action_creation(self):
        """Test creating a CallToolAction."""
        action = CallToolAction(tool_name="echo", arguments={"message": "hello"})
        assert action.type == "call_tool"
        assert action.tool_name == "echo"
        assert action.arguments["message"] == "hello"

    def test_call_tool_action_default_arguments(self):
        """Test CallToolAction has empty dict as default arguments."""
        action = CallToolAction(tool_name="list")
        assert action.arguments == {}

    def test_call_tool_requires_tool_name(self):
        """Test CallToolAction requires tool_name."""
        with pytest.raises(ValidationError):
            CallToolAction()


class TestListToolsObservation:
    """Tests for ListToolsObservation."""

    def test_list_tools_observation_creation(self):
        """Test creating a ListToolsObservation."""
        tools = [
            Tool(name="echo", description="Echo message", input_schema={}),
            Tool(name="greet", description="Greet user", input_schema={}),
        ]
        obs = ListToolsObservation(tools=tools)
        assert len(obs.tools) == 2
        assert obs.tools[0].name == "echo"
        assert obs.done is False  # Default from Observation

    def test_list_tools_observation_empty(self):
        """Test ListToolsObservation with no tools."""
        obs = ListToolsObservation(tools=[])
        assert obs.tools == []


class TestCallToolObservation:
    """Tests for CallToolObservation."""

    def test_call_tool_observation_success(self):
        """Test CallToolObservation for successful call."""
        obs = CallToolObservation(
            tool_name="echo",
            result={"message": "hello", "length": 5},
        )
        assert obs.tool_name == "echo"
        assert obs.result["message"] == "hello"
        assert obs.error is None

    def test_call_tool_observation_with_error(self):
        """Test CallToolObservation with error."""
        obs = CallToolObservation(
            tool_name="broken_tool",
            result=None,
            error=ToolError(
                error_type=ToolErrorType.EXECUTION_ERROR,
                message="Tool crashed",
            ),
        )
        assert obs.tool_name == "broken_tool"
        assert obs.error is not None
        assert obs.error.error_type == ToolErrorType.EXECUTION_ERROR


class TestWSMCPMessage:
    """Tests for WebSocket MCP messages."""

    def test_ws_mcp_message_creation(self):
        """Test creating a WSMCPMessage."""
        msg = WSMCPMessage(data={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        assert msg.type == "mcp"
        assert msg.data["method"] == "tools/list"

    def test_ws_mcp_response_creation(self):
        """Test creating a WSMCPResponse."""
        response = WSMCPResponse(
            data={"jsonrpc": "2.0", "result": {"tools": []}, "id": 1}
        )
        assert response.type == "mcp"
        assert response.data["result"]["tools"] == []


class TestReservedToolNames:
    """Tests for reserved tool names."""

    def test_reserved_names_exist(self):
        """Test that reserved names are defined."""
        assert "reset" in RESERVED_TOOL_NAMES
        assert "step" in RESERVED_TOOL_NAMES
        assert "state" in RESERVED_TOOL_NAMES
        assert "close" in RESERVED_TOOL_NAMES

    def test_reserved_names_is_frozenset(self):
        """Test that reserved names cannot be modified."""
        assert isinstance(RESERVED_TOOL_NAMES, frozenset)


class TestMCPRequest:
    """Tests for HTTP MCP request type (RFC 003)."""

    def test_mcp_request_tools_list(self):
        """Test creating an MCP request for tools/list."""
        request = MCPRequest(method="tools/list", id=1)
        assert request.jsonrpc == "2.0"
        assert request.method == "tools/list"
        assert request.id == 1
        assert request.params is None

    def test_mcp_request_tools_call(self):
        """Test creating an MCP request for tools/call."""
        request = MCPRequest(
            method="tools/call",
            params={"name": "echo", "arguments": {"message": "hello"}},
            id=2,
        )
        assert request.jsonrpc == "2.0"
        assert request.method == "tools/call"
        assert request.params["name"] == "echo"
        assert request.params["arguments"]["message"] == "hello"
        assert request.id == 2

    def test_mcp_request_without_id(self):
        """Test creating an MCP request without id (notification style)."""
        request = MCPRequest(method="tools/list")
        assert request.jsonrpc == "2.0"
        assert request.method == "tools/list"
        assert request.id is None

    def test_mcp_request_requires_method(self):
        """Test that MCPRequest requires method field."""
        with pytest.raises(ValidationError):
            MCPRequest()  # Missing method

    def test_mcp_request_serialization(self):
        """Test MCPRequest serialization to dict."""
        request = MCPRequest(method="tools/list", id=1)
        data = request.model_dump()
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "tools/list"
        assert data["id"] == 1


class TestMCPResponse:
    """Tests for HTTP MCP response type (RFC 003)."""

    def test_mcp_response_success(self):
        """Test creating a successful MCP response."""
        response = MCPResponse(result={"tools": []}, id=1)
        assert response.jsonrpc == "2.0"
        assert response.result == {"tools": []}
        assert response.error is None
        assert response.id == 1

    def test_mcp_response_error(self):
        """Test creating an error MCP response."""
        response = MCPResponse(
            error={"code": -32601, "message": "Method not found"},
            id=1,
        )
        assert response.jsonrpc == "2.0"
        assert response.result is None
        assert response.error["code"] == -32601
        assert "Method not found" in response.error["message"]
        assert response.id == 1

    def test_mcp_response_without_id(self):
        """Test creating an MCP response without id."""
        response = MCPResponse(result="Hello, World!")
        assert response.jsonrpc == "2.0"
        assert response.result == "Hello, World!"
        assert response.id is None

    def test_mcp_response_serialization(self):
        """Test MCPResponse serialization to dict."""
        response = MCPResponse(result={"tools": [{"name": "echo"}]}, id=1)
        data = response.model_dump()
        assert data["jsonrpc"] == "2.0"
        assert data["result"]["tools"][0]["name"] == "echo"
        assert data["id"] == 1
