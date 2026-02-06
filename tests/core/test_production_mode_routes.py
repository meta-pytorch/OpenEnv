# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for production mode routes in OpenEnv.

This file combines two aspects of production mode:

1. Route restrictions (from main): Tests that production mode blocks simulation control
   endpoints (/reset, /step, /state) while allowing safe endpoints. This is a critical
   security boundary: production environments should only expose MCP tools, not simulation
   controls that manipulate time and causality.

2. Direct MCP API access (from issue #347): Per RFC 003, environments should expose both:
   - Training/Eval API: step() for RL training (includes reward computation, state tracking)
   - Production API: Direct MCP endpoints for inference (bypasses step(), no rewards)

Test coverage:
- Production mode disables /reset, /step, /state endpoints (returns 404 or 405)
- Production mode allows /health, /schema, /metadata, /ws endpoints
- Direct MCP JSON-RPC endpoints work (tools/list, tools/call)
- WebSocket MCP message handling
- HTTP POST /mcp endpoint for MCP JSON-RPC
- Production mode bypasses step() overhead
- Proper error responses for invalid MCP requests
"""

import sys
from pathlib import Path
import json
import pytest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.mcp_types import (
    RESERVED_TOOL_NAMES,
)


# ============================================================================
# Test Fixtures - Minimal Environment for Testing
# ============================================================================


class MinimalAction(Action):
    """Minimal action for testing."""

    message: str


class MinimalObservation(Observation):
    """Minimal observation for testing."""

    response: str
    reward: float | None = None
    done: bool = False


class MinimalState(State):
    """Minimal state for testing."""

    step_count: int = 0


class MinimalEnvironment(Environment):
    """Minimal environment implementation for testing server modes."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, **kwargs) -> MinimalObservation:
        """Reset the environment."""
        return MinimalObservation(response="reset", reward=None, done=False)

    def step(self, action: MinimalAction) -> MinimalObservation:
        """Execute an action."""
        return MinimalObservation(
            response=f"echo: {action.message}", reward=1.0, done=False
        )

    @property
    def state(self) -> MinimalState:
        """Return current state."""
        return MinimalState(step_count=0)

    def close(self) -> None:
        """Cleanup resources."""
        pass


@pytest.fixture
def production_mode_app() -> FastAPI:
    """
    Create a FastAPI app with production mode enabled.

    In production mode, /reset, /step, /state should NOT be registered.
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=MinimalEnvironment,
        action_cls=MinimalAction,
        observation_cls=MinimalObservation,
    )
    # TODO: Once production mode is implemented, pass mode="production" here
    # For now, this will fail because the feature doesn't exist yet
    server.register_routes(app, mode="production")
    return app


@pytest.fixture
def simulation_mode_app() -> FastAPI:
    """
    Create a FastAPI app with simulation mode (default).

    In simulation mode, all endpoints including /reset, /step, /state are available.
    """
    app = FastAPI()
    server = HTTPEnvServer(
        env=MinimalEnvironment,
        action_cls=MinimalAction,
        observation_cls=MinimalObservation,
    )
    # Default mode should be simulation
    server.register_routes(app)
    return app


# ============================================================================
# Production Mode Route Restriction Tests (from main)
# ============================================================================


class TestProductionModeRouteRestrictions:
    """Test that production mode hides simulation control endpoints."""

    def test_production_mode_blocks_reset_endpoint(self, production_mode_app):
        """Test that /reset returns 404 or 405 in production mode."""
        client = TestClient(production_mode_app)

        response = client.post("/reset", json={})

        # Should return 404 (Not Found) or 405 (Method Not Allowed)
        assert response.status_code in [404, 405], (
            f"Expected 404 or 405, got {response.status_code}. "
            "Production mode should not expose /reset endpoint."
        )

    def test_production_mode_blocks_step_endpoint(self, production_mode_app):
        """Test that /step returns 404 or 405 in production mode."""
        client = TestClient(production_mode_app)

        response = client.post("/step", json={"action": {"message": "test"}})

        # Should return 404 (Not Found) or 405 (Method Not Allowed)
        assert response.status_code in [404, 405], (
            f"Expected 404 or 405, got {response.status_code}. "
            "Production mode should not expose /step endpoint."
        )

    def test_production_mode_blocks_state_endpoint(self, production_mode_app):
        """Test that /state returns 404 or 405 in production mode."""
        client = TestClient(production_mode_app)

        response = client.get("/state")

        # Should return 404 (Not Found) or 405 (Method Not Allowed)
        assert response.status_code in [404, 405], (
            f"Expected 404 or 405, got {response.status_code}. "
            "Production mode should not expose /state endpoint."
        )


# ============================================================================
# Production Mode Still Allows Safe Endpoints
# ============================================================================


class TestProductionModeAllowsSafeEndpoints:
    """Test that production mode still exposes safe, non-simulation endpoints."""

    def test_production_mode_allows_health_endpoint(self, production_mode_app):
        """Test that /health is still available in production mode."""
        client = TestClient(production_mode_app)

        response = client.get("/health")

        assert response.status_code == 200, (
            "Production mode should still expose /health for monitoring"
        )
        assert response.json()["status"] == "healthy"

    def test_production_mode_allows_schema_endpoint(self, production_mode_app):
        """Test that /schema is still available in production mode."""
        client = TestClient(production_mode_app)

        response = client.get("/schema")

        assert response.status_code == 200, (
            "Production mode should still expose /schema for introspection"
        )
        # Should have action, observation, state schemas
        data = response.json()
        assert "action" in data
        assert "observation" in data
        assert "state" in data

    def test_production_mode_allows_metadata_endpoint(self, production_mode_app):
        """Test that /metadata is still available in production mode."""
        client = TestClient(production_mode_app)

        response = client.get("/metadata")

        assert response.status_code == 200, (
            "Production mode should still expose /metadata for environment info"
        )

    def test_production_mode_allows_websocket_endpoint(self, production_mode_app):
        """Test that /ws WebSocket is still available in production mode."""
        client = TestClient(production_mode_app)

        # WebSocket connection test - we expect it to accept the connection
        # We don't test the full WebSocket protocol here, just that it's registered
        try:
            with client.websocket_connect("/ws") as websocket:
                # If we get here, the endpoint is registered
                # We can close immediately
                websocket.close()
                assert True, "WebSocket endpoint should be available"
        except Exception as e:
            # If the endpoint doesn't exist, we'll get a 404
            pytest.fail(
                f"WebSocket endpoint should be available in production mode: {e}"
            )


# ============================================================================
# Simulation Mode Allows All Endpoints (Regression Test)
# ============================================================================


class TestSimulationModeAllowsAllEndpoints:
    """Test that simulation mode (default) allows all endpoints."""

    def test_simulation_mode_allows_reset_endpoint(self, simulation_mode_app):
        """Test that /reset works in simulation mode (default behavior)."""
        client = TestClient(simulation_mode_app)

        response = client.post("/reset", json={})

        assert response.status_code == 200, (
            "Simulation mode should expose /reset endpoint"
        )
        data = response.json()
        assert "observation" in data
        assert data["observation"]["response"] == "reset"

    def test_simulation_mode_allows_step_endpoint(self, simulation_mode_app):
        """Test that /step works in simulation mode (default behavior)."""
        client = TestClient(simulation_mode_app)

        response = client.post("/step", json={"action": {"message": "hello"}})

        assert response.status_code == 200, (
            "Simulation mode should expose /step endpoint"
        )
        data = response.json()
        assert "observation" in data
        assert "echo: hello" in data["observation"]["response"]

    def test_simulation_mode_allows_state_endpoint(self, simulation_mode_app):
        """Test that /state works in simulation mode (default behavior)."""
        client = TestClient(simulation_mode_app)

        response = client.get("/state")

        assert response.status_code == 200, (
            "Simulation mode should expose /state endpoint"
        )
        data = response.json()
        assert "step_count" in data
        assert data["step_count"] == 0


# ============================================================================
# Mode Configuration Tests
# ============================================================================


class TestModeConfiguration:
    """Test that mode can be configured via parameter."""

    def test_explicit_production_mode_parameter(self):
        """Test that mode='production' can be passed to register_routes."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=MinimalEnvironment,
            action_cls=MinimalAction,
            observation_cls=MinimalObservation,
        )

        # This should not raise an error
        # The implementation should accept mode parameter
        try:
            server.register_routes(app, mode="production")
        except TypeError as e:
            pytest.fail(f"register_routes should accept mode parameter: {e}")

    def test_explicit_simulation_mode_parameter(self):
        """Test that mode='simulation' can be passed to register_routes."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=MinimalEnvironment,
            action_cls=MinimalAction,
            observation_cls=MinimalObservation,
        )

        # This should not raise an error
        try:
            server.register_routes(app, mode="simulation")
        except TypeError as e:
            pytest.fail(f"register_routes should accept mode parameter: {e}")

    def test_default_mode_is_simulation(self):
        """Test that default mode is 'simulation' for backwards compatibility."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=MinimalEnvironment,
            action_cls=MinimalAction,
            observation_cls=MinimalObservation,
        )
        server.register_routes(app)
        client = TestClient(app)

        # Should have /reset, /step, /state in default mode
        reset_response = client.post("/reset", json={})
        step_response = client.post("/step", json={"action": {"message": "test"}})
        state_response = client.get("/state")

        assert reset_response.status_code == 200, "Default mode should allow /reset"
        assert step_response.status_code == 200, "Default mode should allow /step"
        assert state_response.status_code == 200, "Default mode should allow /state"

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode value raises ValueError."""
        app = FastAPI()
        server = HTTPEnvServer(
            env=MinimalEnvironment,
            action_cls=MinimalAction,
            observation_cls=MinimalObservation,
        )

        with pytest.raises(ValueError) as exc_info:
            server.register_routes(app, mode="invalid_mode")

        assert "mode" in str(exc_info.value).lower()
        assert "production" in str(exc_info.value).lower()
        assert "simulation" in str(exc_info.value).lower()


# ============================================================================
# Security Boundary Tests
# ============================================================================


class TestProductionModeSecurityBoundary:
    """
    Test that production mode enforces the security boundary.

    The key invariant: In production, agents cannot control time/causality.
    """

    def test_production_mode_prevents_reset_manipulation(self, production_mode_app):
        """
        Test that production mode prevents environment reset.

        In production, we can't reset the real world - time only moves forward.
        """
        client = TestClient(production_mode_app)

        # Try to reset (should fail)
        response = client.post("/reset", json={"seed": 42})

        assert response.status_code in [404, 405], (
            "Production mode must not allow reset - can't reset the real world"
        )

    def test_production_mode_prevents_state_inspection(self, production_mode_app):
        """
        Test that production mode prevents arbitrary state inspection.

        State inspection is a simulation concept - in prod we only observe via tools.
        """
        client = TestClient(production_mode_app)

        response = client.get("/state")

        assert response.status_code in [404, 405], (
            "Production mode should not expose internal state directly"
        )

    def test_production_mode_prevents_direct_step(self, production_mode_app):
        """
        Test that production mode prevents direct step calls.

        In production, agents interact via MCP tools, not direct step() calls.
        """
        client = TestClient(production_mode_app)

        response = client.post("/step", json={"action": {"message": "test"}})

        assert response.status_code in [404, 405], (
            "Production mode should not allow direct step() - use MCP tools instead"
        )


# ============================================================================
# Direct MCP API Access Tests (from issue #347)
# ============================================================================


# =============================================================================
# Test Fixtures - MCP Endpoints
# =============================================================================


@pytest.fixture
def mock_fastmcp_server():
    """Create a mock FastMCP server for testing."""
    from fastmcp import FastMCP

    mcp = FastMCP("test-server")

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool
    def greet(name: str) -> str:
        """Greet a person."""
        return f"Hello, {name}!"

    return mcp


@pytest.fixture
def app(mock_fastmcp_server):
    """Create FastAPI app with MCP endpoints."""
    # This should FAIL because MCP endpoints are not implemented yet
    from openenv.core.env_server.http_server import create_fastapi_app
    from openenv.core.env_server.mcp_environment import MCPEnvironment

    class TestMCPEnv(MCPEnvironment):
        def __init__(self):
            super().__init__(mock_fastmcp_server)
            self._state = {"step_count": 0}

        def reset(self, **kwargs):
            self._state = {"step_count": 0}
            return Observation(done=False, reward=0.0)

        def _step_impl(self, action, **kwargs):
            self._state["step_count"] += 1
            return Observation(done=False, reward=0.0)

        @property
        def state(self):
            from openenv.core.env_server.types import State

            return State(step_count=self._state["step_count"])

    return create_fastapi_app(
        env=TestMCPEnv,
        action_cls=None,
        observation_cls=None,
    )


# =============================================================================
# HTTP /mcp Endpoint Tests
# =============================================================================


class TestHTTPMCPEndpoint:
    """Tests for HTTP POST /mcp endpoint (JSON-RPC)."""

    def test_mcp_endpoint_exists(self, app):
        """Test /mcp endpoint is exposed."""
        # This should FAIL because /mcp endpoint doesn't exist yet
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        )

        assert response.status_code == 200

    def test_mcp_tools_list_via_http(self, app):
        """Test tools/list via HTTP /mcp endpoint."""
        # This should FAIL because tools/list handling is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) > 0

    def test_mcp_tools_call_via_http(self, app):
        """Test tools/call via HTTP /mcp endpoint."""
        # This should FAIL because tools/call handling is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "add", "arguments": {"a": 5, "b": 3}},
                "id": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        assert "result" in data
        # Result should contain the tool's return value
        assert "8" in str(data["result"]) or data["result"] == 8

    def test_mcp_http_bypasses_step_overhead(self, app):
        """Test direct MCP access doesn't call step() or compute rewards."""
        # This should FAIL because direct MCP path is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)

        with patch(
            "openenv.core.env_server.mcp_environment.MCPEnvironment.step"
        ) as mock_step:
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "add", "arguments": {"a": 1, "b": 1}},
                    "id": 3,
                },
            )

            # Verify step() was NOT called (production mode bypasses it)
            mock_step.assert_not_called()
            assert response.status_code == 200

    def test_mcp_http_invalid_method_returns_error(self, app):
        """Test invalid MCP method returns proper JSON-RPC error."""
        # This should FAIL because error handling is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "invalid/method", "id": 4}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 4
        assert "error" in data
        assert data["error"]["code"] == -32601  # Method not found

    def test_mcp_http_missing_jsonrpc_version(self, app):
        """Test request without jsonrpc version returns error."""
        # This should FAIL because validation is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post("/mcp", json={"method": "tools/list", "id": 5})

        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data

    def test_mcp_http_no_reset_required(self, app):
        """Test MCP endpoints work without calling reset() first."""
        # This should FAIL if reset() is required (it shouldn't be)
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Call tools/list without reset
        response = client.post(
            "/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 6}
        )

        assert response.status_code == 200
        data = response.json()
        assert "tools" in data["result"]


# =============================================================================
# WebSocket MCP Tests
# =============================================================================


class TestWebSocketMCP:
    """Tests for WebSocket MCP message handling."""

    def test_websocket_mcp_message_type(self, app):
        """Test WebSocket accepts 'mcp' message type."""
        # This should FAIL because MCP message handling is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Send MCP message via WebSocket
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    }
                )
            )

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert response["data"]["jsonrpc"] == "2.0"

    def test_websocket_mcp_tools_list(self, app):
        """Test tools/list via WebSocket MCP message."""
        # This should FAIL because WebSocket MCP is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    }
                )
            )

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert "tools" in response["data"]["result"]

    def test_websocket_mcp_tools_call(self, app):
        """Test tools/call via WebSocket MCP message."""
        # This should FAIL because WebSocket MCP is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "greet",
                                "arguments": {"name": "Production"},
                            },
                            "id": 2,
                        },
                    }
                )
            )

            response_text = websocket.receive_text()
            response = json.loads(response_text)

            assert response["type"] == "mcp"
            assert "Production" in str(response["data"]["result"])

    def test_websocket_mcp_interleaved_with_step(self, app):
        """Test WebSocket can handle both MCP and step() messages."""
        # This should FAIL because mixed message handling is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # First, use step() API
            websocket.send_text(json.dumps({"type": "reset", "data": {}}))
            response1 = websocket.receive_text()
            assert json.loads(response1)["type"] == "observation"

            # Then use MCP API directly
            websocket.send_text(
                json.dumps(
                    {
                        "type": "mcp",
                        "data": {"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                    }
                )
            )
            response2 = websocket.receive_text()
            mcp_response = json.loads(response2)

            assert mcp_response["type"] == "mcp"
            assert "tools" in mcp_response["data"]["result"]


# =============================================================================
# Reserved Tool Names Tests
# =============================================================================


class TestReservedToolNames:
    """Tests for reserved tool name validation."""

    def test_reserved_names_constant_exists(self):
        """Test RESERVED_TOOL_NAMES is defined."""
        # This should PASS as it's already defined in mcp_types.py
        assert RESERVED_TOOL_NAMES is not None
        assert isinstance(RESERVED_TOOL_NAMES, frozenset)

    def test_reserved_names_include_env_methods(self):
        """Test reserved names include environment methods."""
        # This should PASS as it's already defined
        assert "reset" in RESERVED_TOOL_NAMES
        assert "step" in RESERVED_TOOL_NAMES
        assert "state" in RESERVED_TOOL_NAMES
        assert "close" in RESERVED_TOOL_NAMES

    def test_mcp_server_rejects_reserved_tool_names(self):
        """Test MCP server validation rejects reserved tool names."""
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @mcp.tool
        def reset() -> str:
            """This uses a reserved name."""
            return "should not work"

        from openenv.core.env_server.mcp_environment import MCPEnvironment

        # Use a concrete subclass to test validation
        class TestMCPEnv(MCPEnvironment):
            def reset(self, **kwargs):
                return Observation(done=False, reward=0.0)

            def _step_impl(self, action, **kwargs):
                return Observation(done=False, reward=0.0)

            @property
            def state(self):
                from openenv.core.env_server.types import State

                return State(step_count=0)

        with pytest.raises(ValueError) as exc_info:
            TestMCPEnv(mcp)

        assert "reserved" in str(exc_info.value).lower()
        assert "reset" in str(exc_info.value)


# =============================================================================
# Performance Tests
# =============================================================================


class TestProductionModePerformance:
    """Tests verifying production mode is optimized for inference."""

    def test_production_mode_no_reward_in_response(self, app):
        """Test production MCP mode returns tool result without reward."""
        from starlette.testclient import TestClient

        client = TestClient(app)

        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "add", "arguments": {"a": 1, "b": 1}},
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # MCP response is pure JSON-RPC - no reward field
        assert "reward" not in data

    def test_production_mode_no_state_tracking(self, app):
        """Test production MCP mode doesn't track episode state."""
        # This should FAIL because production mode optimization is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Get initial state
        state_response = client.get("/state")
        initial_step_count = state_response.json()["step_count"]

        # Call tool via MCP
        client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "add", "arguments": {"a": 1, "b": 1}},
                "id": 1,
            },
        )

        # Verify step count didn't increment (production mode bypasses step tracking)
        state_response = client.get("/state")
        final_step_count = state_response.json()["step_count"]

        assert final_step_count == initial_step_count


# =============================================================================
# Client Integration Tests
# =============================================================================


class TestMCPClientProductionMode:
    """Tests for MCP client using production mode."""

    async def test_mcp_client_can_use_production_endpoints(self):
        """Test MCPToolClient can use production MCP endpoints directly."""
        # This should FAIL because client doesn't expose production mode option
        from openenv.core.mcp_client import MCPToolClient

        client = MCPToolClient(base_url="http://localhost:8000")

        # Client should have option to use production mode (bypasses step())
        assert hasattr(client, "use_production_mode")

        client.use_production_mode = True

        # Calling list_tools() should use /mcp endpoint, not step()
        with patch.object(client, "step") as mock_step:
            tools = await client.list_tools()

            # step() should NOT be called in production mode
            mock_step.assert_not_called()
            assert len(tools) >= 0

    async def test_client_production_mode_uses_http_mcp_endpoint(self):
        """Test client in production mode uses HTTP /mcp endpoint."""
        # This should FAIL because production mode routing is not implemented
        from openenv.core.mcp_client import MCPToolClient

        client = MCPToolClient(base_url="http://localhost:8000")
        client.use_production_mode = True

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {
                "jsonrpc": "2.0",
                "result": {"tools": []},
                "id": 1,
            }

            await client.list_tools()

            # Verify /mcp endpoint was called, not /step
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/mcp" in call_args[0][0]


# =============================================================================
# Error Response Tests
# =============================================================================


class TestMCPErrorResponses:
    """Tests for proper MCP JSON-RPC error responses."""

    def test_invalid_json_returns_parse_error(self, app):
        """Test malformed JSON returns JSON-RPC parse error."""
        # This should FAIL because error handling is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post("/mcp", data="not valid json")

        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == -32700  # Parse error

    def test_missing_params_returns_invalid_params(self, app):
        """Test missing required params returns invalid params error."""
        # This should FAIL because validation is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    # Missing 'name' field
                    "arguments": {"a": 1}
                },
                "id": 1,
            },
        )

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32602  # Invalid params

    def test_nonexistent_tool_returns_error(self, app):
        """Test calling non-existent tool returns proper error."""
        # This should FAIL because error handling is not implemented
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "nonexistent_tool", "arguments": {}},
                "id": 1,
            },
        )

        data = response.json()
        assert "error" in data or "result" in data
        # Should indicate tool not found
