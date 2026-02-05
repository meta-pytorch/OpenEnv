# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for production mode route restrictions.

These tests verify that when the HTTP server is running in production mode,
simulation control endpoints (/reset, /step, /state) are NOT exposed.
This is a critical security boundary: production environments should only
expose MCP tools, not simulation controls that manipulate time and causality.

Test coverage:
- Production mode disables /reset endpoint (returns 404 or 405)
- Production mode disables /step endpoint (returns 404 or 405)
- Production mode disables /state endpoint (returns 404 or 405)
- Production mode still allows /health endpoint
- Production mode still allows /schema endpoint
- Production mode still allows /metadata endpoint
- Production mode still allows /ws WebSocket endpoint
- Simulation mode (default) allows all endpoints
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State


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
# Production Mode Route Restriction Tests
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
