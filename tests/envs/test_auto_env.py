# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for AutoEnv and AutoAction
======================================

Tests cover:
1. AutoEnv factory methods (from_hub, get_env_class, get_env_info, list_environments)
2. AutoAction factory methods (from_hub, from_env, get_action_info, list_actions)
3. Error handling for unknown environments
4. Name normalization and suggestions
5. Hub URL detection and handling
6. Integration with the discovery system
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Type

from openenv.auto.auto_env import AutoEnv
from openenv.auto.auto_action import AutoAction
from openenv.auto._discovery import (
    EnvironmentInfo,
    EnvironmentDiscovery,
    get_discovery,
    reset_discovery,
    _normalize_env_name,
    _is_hub_url,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_env_info():
    """Create a mock EnvironmentInfo for testing."""
    return EnvironmentInfo(
        env_key="echo",
        name="echo_env",
        package_name="openenv-echo-env",
        version="0.1.0",
        description="Echo environment for testing",
        client_module_path="echo_env.client",
        client_class_name="EchoEnv",
        action_class_name="EchoAction",
        observation_class_name="EchoObservation",
        default_image="echo-env:latest",
        spec_version=1,
    )


@pytest.fixture
def mock_coding_env_info():
    """Create a mock EnvironmentInfo for coding environment."""
    return EnvironmentInfo(
        env_key="coding",
        name="coding_env",
        package_name="openenv-coding_env",
        version="0.2.0",
        description="Coding environment with Python execution",
        client_module_path="coding_env.client",
        client_class_name="CodingEnv",
        action_class_name="CodeAction",  # Custom name
        observation_class_name="CodeObservation",  # Custom name
        default_image="coding-env:latest",
        spec_version=1,
    )


@pytest.fixture
def mock_discovery(mock_env_info, mock_coding_env_info):
    """Create a mock discovery instance with test environments."""
    discovery = Mock(spec=EnvironmentDiscovery)
    envs = {
        "echo": mock_env_info,
        "coding": mock_coding_env_info,
    }
    discovery.discover.return_value = envs
    discovery.get_environment.side_effect = lambda key: envs.get(key)
    discovery.get_environment_by_name.side_effect = lambda name: envs.get(
        _normalize_env_name(name).replace("_env", "")
    )
    return discovery


@pytest.fixture(autouse=True)
def reset_global_discovery():
    """Reset global discovery before and after each test."""
    reset_discovery()
    yield
    reset_discovery()


# ============================================================================
# AutoEnv Tests
# ============================================================================


class TestAutoEnvInstantiation:
    """Test that AutoEnv cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        """AutoEnv should raise TypeError when instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            AutoEnv()
        
        assert "factory class" in str(exc_info.value).lower()
        assert "AutoEnv.from_hub()" in str(exc_info.value)


class TestAutoEnvGetEnvClass:
    """Test AutoEnv.get_env_class() method."""

    def test_get_env_class_success(self, mock_discovery, mock_env_info):
        """Test getting environment class successfully."""
        # Mock the discovery
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            # Mock the client class
            mock_client_class = Mock()
            mock_env_info.get_client_class = Mock(return_value=mock_client_class)
            
            result = AutoEnv.get_env_class("echo")
            
            assert result is mock_client_class
            mock_env_info.get_client_class.assert_called_once()

    def test_get_env_class_not_found(self, mock_discovery):
        """Test getting unknown environment raises ValueError."""
        mock_discovery.get_environment_by_name.return_value = None
        
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv.get_env_class("nonexistent")
            
            assert "Unknown environment" in str(exc_info.value)

    def test_get_env_class_with_different_name_formats(self, mock_discovery, mock_env_info):
        """Test that different name formats resolve correctly."""
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            mock_client_class = Mock()
            mock_env_info.get_client_class = Mock(return_value=mock_client_class)
            
            # All these should work
            for name in ["echo", "echo-env", "echo_env"]:
                mock_discovery.get_environment_by_name.return_value = mock_env_info
                result = AutoEnv.get_env_class(name)
                assert result is mock_client_class


class TestAutoEnvGetEnvInfo:
    """Test AutoEnv.get_env_info() method."""

    def test_get_env_info_success(self, mock_discovery, mock_env_info):
        """Test getting environment info successfully."""
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            mock_discovery.get_environment_by_name.return_value = mock_env_info
            
            info = AutoEnv.get_env_info("echo")
            
            assert info["env_key"] == "echo"
            assert info["name"] == "echo_env"
            assert info["package"] == "openenv-echo-env"
            assert info["version"] == "0.1.0"
            assert info["description"] == "Echo environment for testing"
            assert info["env_class"] == "EchoEnv"
            assert info["action_class"] == "EchoAction"
            assert info["observation_class"] == "EchoObservation"
            assert info["module"] == "echo_env.client"
            assert info["default_image"] == "echo-env:latest"

    def test_get_env_info_not_found(self, mock_discovery):
        """Test getting info for unknown environment raises ValueError."""
        mock_discovery.get_environment_by_name.return_value = None
        
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv.get_env_info("nonexistent")
            
            assert "Unknown environment" in str(exc_info.value)


class TestAutoEnvListEnvironments:
    """Test AutoEnv.list_environments() method."""

    def test_list_environments(self, mock_discovery, capsys):
        """Test listing environments prints formatted output."""
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            AutoEnv.list_environments()
        
        captured = capsys.readouterr()
        # Should call discovery.list_environments()
        mock_discovery.list_environments.assert_called_once()


class TestAutoEnvFromName:
    """Test AutoEnv.from_hub() method."""

    def test_from_hub_unknown_env_with_suggestions(self, mock_discovery):
        """Test that unknown environment provides suggestions."""
        mock_discovery.get_environment_by_name.return_value = None
        mock_discovery.discover.return_value = {
            "echo": Mock(),
            "coding": Mock(),
        }
        
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv.from_hub("ech")  # Close to "echo"
            
            error_msg = str(exc_info.value)
            assert "Unknown environment" in error_msg or "ech" in error_msg
            # Should suggest similar names
            assert "echo" in error_msg.lower() or "available" in error_msg.lower()

    def test_from_hub_no_envs_available(self, mock_discovery):
        """Test error message when no environments are installed."""
        mock_discovery.get_environment_by_name.return_value = None
        mock_discovery.discover.return_value = {}
        
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoEnv.from_hub("anyenv")
            
            error_msg = str(exc_info.value)
            assert "No OpenEnv environments found" in error_msg
            assert "pip install" in error_msg

    def test_from_hub_with_base_url(self, mock_discovery, mock_env_info):
        """Test from_hub with explicit base_url."""
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        
        # Mock the client class
        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_env_info.get_client_class = Mock(return_value=mock_client_class)
        
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            result = AutoEnv.from_hub("echo", base_url="http://localhost:8000")
            
            assert result is mock_client_instance
            mock_client_class.assert_called_once_with(
                base_url="http://localhost:8000",
                provider=None
            )


class TestAutoEnvHubDetection:
    """Test AutoEnv Hub URL detection and handling."""

    def test_resolve_space_url(self):
        """Test resolving HuggingFace Space URL."""
        url = AutoEnv._resolve_space_url("wukaixingxp/coding-env-test")
        assert url == "https://wukaixingxp-coding-env-test.hf.space"

    def test_resolve_space_url_from_full_url(self):
        """Test resolving from full HuggingFace URL."""
        url = AutoEnv._resolve_space_url("https://huggingface.co/wukaixingxp/coding-env-test")
        assert url == "https://wukaixingxp-coding-env-test.hf.space"


# ============================================================================
# AutoAction Tests
# ============================================================================


class TestAutoActionInstantiation:
    """Test that AutoAction cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        """AutoAction should raise TypeError when instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            AutoAction()
        
        assert "factory class" in str(exc_info.value).lower()
        assert "AutoAction.from_hub()" in str(exc_info.value)


class TestAutoActionFromName:
    """Test AutoAction.from_hub() method."""

    def test_from_hub_success(self, mock_discovery, mock_env_info):
        """Test getting action class successfully."""
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            mock_discovery.get_environment_by_name.return_value = mock_env_info
            
            # Mock the action class
            mock_action_class = Mock()
            mock_env_info.get_action_class = Mock(return_value=mock_action_class)
            
            result = AutoAction.from_hub("echo")
            
            assert result is mock_action_class
            mock_env_info.get_action_class.assert_called_once()

    def test_from_hub_not_found(self, mock_discovery):
        """Test getting unknown action raises ValueError."""
        mock_discovery.get_environment_by_name.return_value = None
        mock_discovery.discover.return_value = {}
        
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoAction.from_hub("nonexistent")
            
            error_msg = str(exc_info.value)
            assert "No OpenEnv environments found" in error_msg

    def test_from_hub_with_suggestions(self, mock_discovery):
        """Test that unknown action provides suggestions."""
        mock_discovery.get_environment_by_name.return_value = None
        mock_discovery.discover.return_value = {
            "echo": Mock(),
            "coding": Mock(),
        }
        
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoAction.from_hub("ech")  # Close to "echo"
            
            error_msg = str(exc_info.value)
            assert "Unknown environment" in error_msg or "ech" in error_msg

    def test_from_hub_with_different_formats(self, mock_discovery, mock_env_info):
        """Test that different name formats work."""
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            mock_action_class = Mock()
            mock_env_info.get_action_class = Mock(return_value=mock_action_class)
            
            # All these should work
            for name in ["echo", "echo-env", "echo_env"]:
                mock_discovery.get_environment_by_name.return_value = mock_env_info
                result = AutoAction.from_hub(name)
                assert result is mock_action_class


class TestAutoActionFromEnv:
    """Test AutoAction.from_env() method (alias for from_hub)."""

    def test_from_env_is_alias(self, mock_discovery, mock_env_info):
        """Test that from_env is an alias for from_hub."""
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            mock_discovery.get_environment_by_name.return_value = mock_env_info
            
            mock_action_class = Mock()
            mock_env_info.get_action_class = Mock(return_value=mock_action_class)
            
            result = AutoAction.from_env("echo")
            
            assert result is mock_action_class


class TestAutoActionGetActionInfo:
    """Test AutoAction.get_action_info() method."""

    def test_get_action_info_success(self, mock_discovery, mock_env_info):
        """Test getting action info successfully."""
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            mock_discovery.get_environment_by_name.return_value = mock_env_info
            
            info = AutoAction.get_action_info("echo")
            
            assert info["env_key"] == "echo"
            assert info["env_name"] == "echo_env"
            assert info["package"] == "openenv-echo-env"
            assert info["action_class"] == "EchoAction"
            assert info["observation_class"] == "EchoObservation"
            assert info["module"] == "echo_env.client"

    def test_get_action_info_with_custom_names(self, mock_discovery, mock_coding_env_info):
        """Test getting action info with custom class names."""
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            mock_discovery.get_environment_by_name.return_value = mock_coding_env_info
            
            info = AutoAction.get_action_info("coding")
            
            assert info["action_class"] == "CodeAction"
            assert info["observation_class"] == "CodeObservation"

    def test_get_action_info_not_found(self, mock_discovery):
        """Test getting info for unknown environment raises ValueError."""
        mock_discovery.get_environment_by_name.return_value = None
        
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            with pytest.raises(ValueError) as exc_info:
                AutoAction.get_action_info("nonexistent")
            
            assert "Unknown environment" in str(exc_info.value)


class TestAutoActionListActions:
    """Test AutoAction.list_actions() method."""

    def test_list_actions_with_envs(self, mock_discovery, mock_env_info, mock_coding_env_info, capsys):
        """Test listing actions prints formatted output."""
        mock_discovery.discover.return_value = {
            "echo": mock_env_info,
            "coding": mock_coding_env_info,
        }
        
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            AutoAction.list_actions()
        
        captured = capsys.readouterr()
        assert "Available Action Classes" in captured.out
        assert "echo" in captured.out
        assert "EchoAction" in captured.out
        assert "coding" in captured.out
        assert "CodeAction" in captured.out
        assert "Total: 2 action classes" in captured.out

    def test_list_actions_empty(self, mock_discovery, capsys):
        """Test listing when no environments are found."""
        mock_discovery.discover.return_value = {}
        
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            AutoAction.list_actions()
        
        captured = capsys.readouterr()
        assert "No OpenEnv environments found" in captured.out
        assert "pip install openenv-" in captured.out


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestNormalizeEnvName:
    """Test _normalize_env_name helper function."""

    def test_simple_name(self):
        """Test normalizing simple names."""
        assert _normalize_env_name("echo") == "echo_env"
        assert _normalize_env_name("coding") == "coding_env"

    def test_name_with_hyphen_suffix(self):
        """Test normalizing names with -env suffix."""
        assert _normalize_env_name("echo-env") == "echo_env"
        assert _normalize_env_name("coding-env") == "coding_env"

    def test_name_with_underscore_suffix(self):
        """Test normalizing names with _env suffix."""
        assert _normalize_env_name("echo_env") == "echo_env"
        assert _normalize_env_name("coding_env") == "coding_env"

    def test_name_with_hyphens(self):
        """Test normalizing names with hyphens."""
        assert _normalize_env_name("browser-gym") == "browser_gym_env"
        assert _normalize_env_name("sumo-rl") == "sumo_rl_env"


class TestIsHubUrl:
    """Test _is_hub_url helper function."""

    def test_org_repo_pattern(self):
        """Test Hub detection with org/repo pattern."""
        assert _is_hub_url("meta-pytorch/coding-env") is True
        assert _is_hub_url("myorg/myenv") is True
        assert _is_hub_url("wukaixingxp/echo-env-test") is True

    def test_full_url(self):
        """Test Hub detection with full URL."""
        assert _is_hub_url("https://huggingface.co/meta-pytorch/coding-env") is True
        assert _is_hub_url("huggingface.co/spaces/myenv") is True

    def test_local_names(self):
        """Test that local names are not detected as Hub URLs."""
        assert _is_hub_url("echo") is False
        assert _is_hub_url("coding-env") is False
        assert _is_hub_url("echo_env") is False
        assert _is_hub_url("browsergym") is False


# ============================================================================
# Integration Tests
# ============================================================================


class TestAutoEnvAutoActionIntegration:
    """Test integration between AutoEnv and AutoAction."""

    def test_same_env_resolves_consistently(self, mock_discovery, mock_env_info):
        """Test that AutoEnv and AutoAction resolve the same environment."""
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery), \
             patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            
            mock_discovery.get_environment_by_name.return_value = mock_env_info
            
            # Mock classes
            mock_client_class = Mock()
            mock_action_class = Mock()
            mock_env_info.get_client_class = Mock(return_value=mock_client_class)
            mock_env_info.get_action_class = Mock(return_value=mock_action_class)
            
            env_class = AutoEnv.get_env_class("echo")
            action_class = AutoAction.from_hub("echo")
            
            # Both should resolve from the same env_info
            assert env_class is mock_client_class
            assert action_class is mock_action_class

    def test_env_info_matches_action_info(self, mock_discovery, mock_env_info):
        """Test that env info and action info are consistent."""
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery), \
             patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            
            mock_discovery.get_environment_by_name.return_value = mock_env_info
            
            env_info = AutoEnv.get_env_info("echo")
            action_info = AutoAction.get_action_info("echo")
            
            # Should have consistent information
            assert env_info["action_class"] == action_info["action_class"]
            assert env_info["observation_class"] == action_info["observation_class"]
            assert env_info["module"] == action_info["module"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling in AutoEnv and AutoAction."""

    def test_import_error_handling(self, mock_discovery, mock_env_info):
        """Test handling of import errors when loading classes."""
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_env_info.get_client_class = Mock(side_effect=ImportError("Module not found"))
        
        with patch('openenv.auto.auto_env.get_discovery', return_value=mock_discovery):
            with pytest.raises(ImportError) as exc_info:
                AutoEnv.from_hub("echo", base_url="http://localhost:8000")
            
            error_msg = str(exc_info.value)
            assert "Failed to import" in error_msg
            assert "pip install" in error_msg or "reinstall" in error_msg

    def test_action_import_error_handling(self, mock_discovery, mock_env_info):
        """Test handling of import errors when loading action classes."""
        mock_discovery.get_environment_by_name.return_value = mock_env_info
        mock_env_info.get_action_class = Mock(side_effect=ImportError("Module not found"))
        
        with patch('openenv.auto.auto_action.get_discovery', return_value=mock_discovery):
            with pytest.raises(ImportError) as exc_info:
                AutoAction.from_hub("echo")
            
            error_msg = str(exc_info.value)
            assert "Failed to import" in error_msg


class TestNameVariations:
    """Test various name format variations work correctly."""

    @pytest.mark.parametrize("name,expected_key", [
        ("echo", "echo"),
        ("echo-env", "echo"),
        ("echo_env", "echo"),
        ("coding", "coding"),
        ("coding-env", "coding"),
        ("coding_env", "coding"),
        ("browser-gym", "browser_gym"),
        ("browser_gym", "browser_gym"),
        ("sumo-rl", "sumo_rl"),
        ("sumo_rl", "sumo_rl"),
    ])
    def test_name_normalization_variations(self, name, expected_key):
        """Test that various name formats normalize correctly."""
        normalized = _normalize_env_name(name)
        key = normalized.replace("_env", "")
        assert key == expected_key


# ============================================================================
# Real Integration Tests - HuggingFace Space
# ============================================================================
# These tests require network access and connect to real HuggingFace Spaces.
# Run with: pytest -m integration tests/envs/test_auto_env.py
# Or: pytest -m "integration and network" tests/envs/test_auto_env.py


@pytest.mark.integration
@pytest.mark.network
class TestHuggingFaceSpaceIntegration:
    """
    Real integration tests that connect to HuggingFace Spaces.
    
    These tests require:
    - Network access to huggingface.co and *.hf.space
    - The HuggingFace Space to be running and accessible
    
    Run these tests with:
        pytest -m "integration and network" tests/envs/test_auto_env.py -v
    """

    # Test Space URL - this is a real HuggingFace Space
    HF_SPACE_REPO = "wukaixingxp/coding-env-test"

    @pytest.fixture
    def check_space_availability(self):
        """Check if the HuggingFace Space is accessible before running tests."""
        import requests
        
        space_url = AutoEnv._resolve_space_url(self.HF_SPACE_REPO)
        try:
            response = requests.get(f"{space_url}/health", timeout=10)
            if response.status_code != 200:
                pytest.skip(f"HuggingFace Space not accessible at {space_url}")
        except requests.RequestException as e:
            pytest.skip(f"Cannot reach HuggingFace Space: {e}")

    def test_connect_to_hf_space(self, check_space_availability):
        """
        Test connecting to a real HuggingFace Space using AutoEnv.
        
        This test:
        1. Connects to wukaixingxp/coding-env-test Space
        2. Resets the environment
        3. Verifies we get a valid observation
        """
        # Connect to HuggingFace Space
        env = AutoEnv.from_hub(self.HF_SPACE_REPO)
        
        try:
            # Reset the environment
            result = env.reset()
            
            # Verify we got a valid result
            assert result is not None
            assert hasattr(result, 'observation')
            
            print(f"✅ Successfully connected to HuggingFace Space: {self.HF_SPACE_REPO}")
            print(f"   Reset observation: {result.observation}")
        finally:
            # Clean up
            env.close()

    def test_execute_action_on_hf_space(self, check_space_availability):
        """
        Test executing an action on a real HuggingFace Space.
        
        This test:
        1. Connects to wukaixingxp/coding-env-test Space
        2. Gets the action class using AutoAction
        3. Executes Python code
        4. Verifies the output
        """
        # Connect to HuggingFace Space
        env = AutoEnv.from_hub(self.HF_SPACE_REPO)
        
        try:
            # Reset the environment
            env.reset()
            
            # Get action class using AutoAction
            CodeAction = AutoAction.from_hub(self.HF_SPACE_REPO)
            
            # Create and execute action
            action = CodeAction(code="print('Hello from pytest!')")
            result = env.step(action)
            
            # Verify the result
            assert result is not None
            assert hasattr(result, 'observation')
            assert hasattr(result, 'reward')
            assert hasattr(result, 'done')
            
            # Check if stdout contains our message
            if hasattr(result.observation, 'stdout'):
                assert "Hello from pytest!" in result.observation.stdout
                print(f"✅ Code execution successful!")
                print(f"   stdout: {result.observation.stdout}")
            
            print(f"   reward: {result.reward}")
            print(f"   done: {result.done}")
        finally:
            # Clean up
            env.close()

    def test_autoenv_and_autoaction_same_space(self, check_space_availability):
        """
        Test that AutoEnv and AutoAction work together seamlessly.
        
        Verifies that calling both with the same HF Space repo ID
        doesn't cause duplicate downloads or installations.
        """
        # First call - AutoEnv
        env = AutoEnv.from_hub(self.HF_SPACE_REPO)
        
        try:
            # Second call - AutoAction (should use cached package)
            ActionClass = AutoAction.from_hub(self.HF_SPACE_REPO)
            
            # Verify both work
            result = env.reset()
            assert result is not None
            
            # Create an action instance
            action = ActionClass(code="x = 1 + 1")
            step_result = env.step(action)
            
            assert step_result is not None
            print(f"✅ AutoEnv and AutoAction work together correctly")
        finally:
            env.close()

    def test_space_availability_check(self):
        """Test the Space availability check functionality."""
        import requests
        
        # Test with real Space URL
        space_url = AutoEnv._resolve_space_url(self.HF_SPACE_REPO)
        
        # Check availability (this is a real network call)
        try:
            is_available = AutoEnv._check_space_availability(space_url, timeout=10.0)
            print(f"Space {space_url} availability: {is_available}")
            # We don't assert True because the space might be down
        except Exception as e:
            pytest.skip(f"Network error checking Space availability: {e}")


# ============================================================================
# Real Integration Tests - Local Docker
# ============================================================================
# These tests require Docker to be installed and running.
# Run with: pytest -m "integration and docker" tests/envs/test_auto_env.py


@pytest.mark.integration
@pytest.mark.docker
class TestDockerIntegration:
    """
    Real integration tests that start Docker containers.
    
    These tests require:
    - Docker to be installed and running
    - Docker images to be built (e.g., echo-env:latest)
    
    Build the Docker image first:
        cd src/envs/echo_env/server && docker build -t echo-env:latest .
    
    Run these tests with:
        pytest -m "integration and docker" tests/envs/test_auto_env.py -v
    """

    @pytest.fixture
    def check_docker_available(self):
        """Check if Docker is available and the required image exists."""
        import subprocess
        import shutil
        
        # Check if docker command exists
        if not shutil.which("docker"):
            pytest.skip("Docker is not installed")
        
        # Check if Docker daemon is running
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                pytest.skip("Docker daemon is not running")
        except subprocess.TimeoutExpired:
            pytest.skip("Docker daemon not responding")
        except Exception as e:
            pytest.skip(f"Cannot access Docker: {e}")

    @pytest.fixture
    def check_echo_env_image(self, check_docker_available):
        """Check if the echo-env Docker image is available."""
        import subprocess
        
        result = subprocess.run(
            ["docker", "images", "-q", "echo-env:latest"],
            capture_output=True,
            text=True
        )
        
        if not result.stdout.strip():
            pytest.skip(
                "Docker image 'echo-env:latest' not found. "
                "Build it with: cd src/envs/echo_env/server && docker build -t echo-env:latest ."
            )

    def test_autoenv_with_docker_echo_env(self, check_echo_env_image):
        """
        Test AutoEnv with a real Docker container (echo-env).
        
        This test:
        1. Starts an echo-env Docker container using AutoEnv
        2. Sends a message
        3. Verifies the echo response
        4. Cleans up the container
        """
        from envs.echo_env.models import EchoAction
        
        # Start Docker container using AutoEnv
        env = AutoEnv.from_hub("echo", docker_image="echo-env:latest")
        
        try:
            # Reset the environment
            result = env.reset()
            assert result is not None
            assert hasattr(result, 'observation')
            
            print(f"✅ Docker container started successfully")
            print(f"   Reset observation: {result.observation}")
            
            # Send a message
            action = EchoAction(message="Hello from Docker test!")
            step_result = env.step(action)
            
            # Verify the echo
            assert step_result is not None
            assert hasattr(step_result.observation, 'echoed_message')
            assert "Hello from Docker test!" in step_result.observation.echoed_message
            
            print(f"✅ Message echoed successfully")
            print(f"   echoed_message: {step_result.observation.echoed_message}")
        finally:
            # Clean up - this should stop the container
            env.close()

    def test_autoaction_with_docker_echo_env(self, check_echo_env_image):
        """
        Test AutoAction with a real Docker container (echo-env).
        
        This test uses AutoAction to get the action class dynamically.
        """
        # Get the action class using AutoAction
        EchoAction = AutoAction.from_hub("echo")
        
        # Start Docker container using AutoEnv
        env = AutoEnv.from_hub("echo", docker_image="echo-env:latest")
        
        try:
            # Reset
            env.reset()
            
            # Create action using the dynamically loaded class
            action = EchoAction(message="Dynamic action from AutoAction!")
            step_result = env.step(action)
            
            # Verify
            assert step_result is not None
            assert "Dynamic action from AutoAction!" in step_result.observation.echoed_message
            
            print(f"✅ AutoAction with Docker works correctly")
        finally:
            env.close()

    def test_env_info_for_docker_env(self, check_docker_available):
        """Test getting environment info for a Docker-based environment."""
        try:
            info = AutoEnv.get_env_info("echo")
            
            assert info is not None
            assert info["env_key"] == "echo"
            assert info["default_image"] == "echo-env:latest"
            
            print(f"✅ Environment info retrieved successfully")
            print(f"   env_key: {info['env_key']}")
            print(f"   default_image: {info['default_image']}")
            print(f"   env_class: {info['env_class']}")
        except ValueError as e:
            pytest.skip(f"Echo environment not installed: {e}")


# ============================================================================
# Real Integration Tests - Local Server
# ============================================================================
# These tests connect to a local server without Docker


@pytest.mark.integration
class TestLocalServerIntegration:
    """
    Integration tests that connect to a locally running server.
    
    These tests require a server to be running on localhost.
    
    Start a server first:
        cd src && python -m envs.echo_env.server.app
    
    Run these tests with:
        pytest -m integration tests/envs/test_auto_env.py::TestLocalServerIntegration -v
    """

    @pytest.fixture
    def local_echo_server(self):
        """Check if local echo server is running."""
        import requests
        
        base_url = "http://localhost:8000"
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Local echo server not healthy")
            return base_url
        except requests.RequestException:
            pytest.skip(
                "Local echo server not running. "
                "Start it with: cd src && python -m envs.echo_env.server.app"
            )

    def test_autoenv_with_local_server(self, local_echo_server):
        """
        Test AutoEnv connecting to a local server using base_url.
        
        This test:
        1. Connects to localhost:8000 using AutoEnv
        2. Resets the environment
        3. Sends a message
        4. Verifies the response
        """
        # Connect to local server
        env = AutoEnv.from_hub("echo", base_url=local_echo_server)
        
        try:
            # Reset
            result = env.reset()
            assert result is not None
            
            print(f"✅ Connected to local server at {local_echo_server}")
            
            # Get action class
            EchoAction = AutoAction.from_hub("echo")
            
            # Send message
            action = EchoAction(message="Hello local server!")
            step_result = env.step(action)
            
            assert step_result is not None
            assert "Hello local server!" in step_result.observation.echoed_message
            
            print(f"✅ Local server test passed")
            print(f"   echoed_message: {step_result.observation.echoed_message}")
        finally:
            env.close()

    def test_multiple_steps_local_server(self, local_echo_server):
        """Test multiple steps on local server."""
        env = AutoEnv.from_hub("echo", base_url=local_echo_server)
        EchoAction = AutoAction.from_hub("echo")
        
        try:
            env.reset()
            
            messages = ["First message", "Second message", "Third message"]
            
            for i, msg in enumerate(messages):
                action = EchoAction(message=msg)
                result = env.step(action)
                
                assert msg in result.observation.echoed_message
                print(f"✅ Step {i+1}: '{msg}' → '{result.observation.echoed_message}'")
            
            print(f"✅ Multiple steps test passed ({len(messages)} steps)")
        finally:
            env.close()


# ============================================================================
# Test Markers Configuration
# ============================================================================
# Add this to conftest.py or pyproject.toml:
#
# [tool.pytest.ini_options]
# markers = [
#     "integration: mark test as integration test (may require external resources)",
#     "network: mark test as requiring network access",
#     "docker: mark test as requiring Docker",
# ]
