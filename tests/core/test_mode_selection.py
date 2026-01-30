# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for client mode selection mechanism.

These tests verify that clients can select between WebSocket (Gym-style) and MCP modes
through constructor parameters and environment variables. The mode selection determines
which protocol the client uses to communicate with the server:

1. Simulation mode: Uses WSResetMessage, WSStepMessage (standard Gym API)
2. Production mode: Uses WSMCPMessage with JSON-RPC (MCP protocol for tool calling)

Test coverage:
- Mode selection via constructor parameter
- Mode selection via environment variable
- Environment variable precedence over constructor default
- Invalid mode values raise appropriate errors
- Mode switching not allowed after connection
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from openenv.core.generic_client import GenericEnvClient
from openenv.core.mcp_client import MCPToolClient


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def clean_env():
    """Ensure OPENENV_CLIENT_MODE is not set."""
    old_mode = os.environ.pop("OPENENV_CLIENT_MODE", None)
    yield
    if old_mode is not None:
        os.environ["OPENENV_CLIENT_MODE"] = old_mode


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.recv.return_value = '{"type": "response", "data": {}}'
    return ws


# ============================================================================
# Constructor Parameter Mode Selection Tests
# ============================================================================


class TestConstructorModeSelection:
    """Test mode selection via constructor parameter."""

    def test_default_mode_is_simulation(self, clean_env):
        """Test that default mode is 'simulation' when no mode specified."""
        client = GenericEnvClient(base_url="http://localhost:8000")

        # Should have simulation mode set
        assert hasattr(client, "_mode")
        assert client._mode == "simulation"

    def test_explicit_simulation_mode(self, clean_env):
        """Test explicit simulation mode via constructor."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

        assert client._mode == "simulation"

    def test_explicit_production_mode(self, clean_env):
        """Test explicit production mode via constructor."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="production")

        assert client._mode == "production"

    def test_invalid_mode_raises_error(self, clean_env):
        """Test that invalid mode value raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            GenericEnvClient(base_url="http://localhost:8000", mode="invalid_mode")

        assert "mode" in str(exc_info.value).lower()
        assert "simulation" in str(exc_info.value).lower()
        assert "production" in str(exc_info.value).lower()

    def test_case_insensitive_mode(self, clean_env):
        """Test that mode parameter is case-insensitive."""
        client1 = GenericEnvClient(base_url="http://localhost:8000", mode="SIMULATION")
        client2 = GenericEnvClient(base_url="http://localhost:8000", mode="PRODUCTION")

        assert client1._mode == "simulation"
        assert client2._mode == "production"


# ============================================================================
# Environment Variable Mode Selection Tests
# ============================================================================


class TestEnvironmentVariableModeSelection:
    """Test mode selection via OPENENV_CLIENT_MODE environment variable."""

    def test_env_var_simulation_mode(self):
        """Test mode selection via OPENENV_CLIENT_MODE=simulation."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "simulation"}):
            client = GenericEnvClient(base_url="http://localhost:8000")
            assert client._mode == "simulation"

    def test_env_var_production_mode(self):
        """Test mode selection via OPENENV_CLIENT_MODE=production."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "production"}):
            client = GenericEnvClient(base_url="http://localhost:8000")
            assert client._mode == "production"

    def test_env_var_case_insensitive(self):
        """Test that OPENENV_CLIENT_MODE is case-insensitive."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "PRODUCTION"}):
            client = GenericEnvClient(base_url="http://localhost:8000")
            assert client._mode == "production"

    def test_env_var_overrides_default(self):
        """Test that environment variable overrides default mode."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "production"}):
            # No explicit mode in constructor
            client = GenericEnvClient(base_url="http://localhost:8000")
            assert client._mode == "production"

    def test_constructor_overrides_env_var(self):
        """Test that explicit constructor parameter overrides environment variable."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "production"}):
            # Explicit mode in constructor should take precedence
            client = GenericEnvClient(
                base_url="http://localhost:8000", mode="simulation"
            )
            assert client._mode == "simulation"

    def test_invalid_env_var_raises_error(self):
        """Test that invalid OPENENV_CLIENT_MODE raises ValueError."""
        with patch.dict(os.environ, {"OPENENV_CLIENT_MODE": "invalid"}):
            with pytest.raises(ValueError) as exc_info:
                GenericEnvClient(base_url="http://localhost:8000")

            assert "OPENENV_CLIENT_MODE" in str(exc_info.value)
            assert "invalid" in str(exc_info.value).lower()


# ============================================================================
# Mode Behavior Tests
# ============================================================================


class TestModeBehavior:
    """Test that different modes result in different client behavior."""

    @pytest.mark.asyncio
    async def test_simulation_mode_uses_gym_protocol(self, clean_env, mock_websocket):
        """Test that simulation mode uses Gym-style WebSocket messages."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

        with patch.object(client, "_send") as mock_send:
            with patch.object(
                client,
                "_receive",
                return_value={
                    "type": "response",
                    "data": {"observation": {}, "reward": None, "done": False},
                },
            ):
                with patch.object(client, "_ws", mock_websocket):
                    await client.reset()

                    # Should send WSResetMessage format
                    call_args = mock_send.call_args_list
                    reset_call = [
                        call for call in call_args if call[0][0].get("type") == "reset"
                    ]
                    assert len(reset_call) > 0, (
                        "Should send reset message with type='reset'"
                    )

    @pytest.mark.asyncio
    async def test_production_mode_uses_jsonrpc_protocol(
        self, clean_env, mock_websocket
    ):
        """Test that production mode uses JSON-RPC format for tool calls."""
        client = MCPToolClient(base_url="http://localhost:8000", mode="production")

        with patch.object(client, "_send") as mock_send:
            with patch.object(
                client,
                "_receive",
                return_value={
                    "type": "response",
                    "data": {
                        "observation": {"tools": []},
                        "reward": None,
                        "done": False,
                    },
                },
            ):
                with patch.object(client, "_ws", mock_websocket):
                    await client.list_tools()

                    # Should send step message with list_tools action
                    call_args = mock_send.call_args_list
                    step_call = [
                        call for call in call_args if call[0][0].get("type") == "step"
                    ]
                    assert len(step_call) > 0, "Should send message with type='step'"

                    # Check that the action payload is list_tools
                    step_message = step_call[0][0][0]
                    assert "data" in step_message
                    assert step_message["data"].get("type") == "list_tools"


# ============================================================================
# Mode Immutability Tests
# ============================================================================


class TestModeImmutability:
    """Test that mode cannot be changed after client creation."""

    def test_mode_cannot_be_changed_after_creation(self, clean_env):
        """Test that mode attribute is read-only after initialization."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

        # Attempting to change mode should raise AttributeError or have no effect
        with pytest.raises((AttributeError, ValueError)):
            client._mode = "mcp"

    def test_mode_cannot_be_changed_after_connection(self, clean_env):
        """Test that mode cannot be changed after connection is established."""
        client = GenericEnvClient(base_url="http://localhost:8000", mode="simulation")

        with patch.object(client, "_ws", MagicMock()):
            # Mark as connected
            client._ws = MagicMock()

            # Should not allow mode change
            with pytest.raises((AttributeError, ValueError)):
                client._mode = "mcp"


# ============================================================================
# Cross-Client Mode Consistency Tests
# ============================================================================


class TestCrossClientModeConsistency:
    """Test that mode selection works consistently across different client types."""

    def test_generic_client_supports_both_modes(self, clean_env):
        """Test that GenericEnvClient supports both simulation and production modes."""
        ws_client = GenericEnvClient(
            base_url="http://localhost:8000", mode="simulation"
        )
        mcp_client = GenericEnvClient(
            base_url="http://localhost:8000", mode="production"
        )

        assert ws_client._mode == "simulation"
        assert mcp_client._mode == "production"

    def test_mcp_client_defaults_to_production_mode(self, clean_env):
        """Test that MCPToolClient defaults to 'production' mode."""
        client = MCPToolClient(base_url="http://localhost:8000")

        # MCPToolClient should default to production mode
        assert client._mode == "production"

    def test_mcp_client_cannot_use_simulation_mode(self, clean_env):
        """Test that MCPToolClient raises error if simulation mode is requested."""
        with pytest.raises(ValueError) as exc_info:
            MCPToolClient(base_url="http://localhost:8000", mode="simulation")

        assert "MCPToolClient" in str(exc_info.value)
        assert "production" in str(exc_info.value).lower()


# ============================================================================
# Mode Documentation Tests
# ============================================================================


class TestModeDocumentation:
    """Test that mode parameter is properly documented."""

    def test_mode_parameter_in_docstring(self, clean_env):
        """Test that mode parameter is documented in __init__ docstring."""
        # GenericEnvClient should document mode parameter
        docstring = GenericEnvClient.__init__.__doc__

        # Should mention mode in Args section
        assert docstring is not None
        assert "mode" in docstring.lower()

    def test_mode_values_documented(self, clean_env):
        """Test that valid mode values are documented."""
        docstring = GenericEnvClient.__init__.__doc__

        # Should document both simulation and production modes
        assert "simulation" in docstring.lower()
        assert "production" in docstring.lower()
