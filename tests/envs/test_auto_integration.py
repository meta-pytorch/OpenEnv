# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for AutoEnv and AutoAction
=============================================

Tests the full integration of package-based discovery with AutoEnv/AutoAction.

These tests use the actual installed packages (echo_env, coding_env) to verify
the complete flow works end-to-end.
"""

import pytest
from envs import AutoEnv, AutoAction
from envs._discovery import reset_discovery


class TestAutoEnvIntegration:
    """Test AutoEnv integration with package discovery."""

    def setup_method(self):
        """Reset discovery before each test to ensure clean state."""
        reset_discovery()

    def test_auto_env_get_env_class(self):
        """Test getting environment class by name."""
        # Test with echo environment (should work if echo_env package is installed)
        try:
            EchoEnv = AutoEnv.get_env_class("echo")
            assert EchoEnv.__name__ == "EchoEnv"
            assert "echo_env.client" in EchoEnv.__module__
        except (ValueError, ImportError) as e:
            # If package not installed or can't be imported, skip test
            pytest.skip(f"echo_env package not properly installed: {e}")

    def test_auto_env_get_env_class_flexible_naming(self):
        """Test flexible name matching."""
        try:
            # All these should work
            EchoEnv1 = AutoEnv.get_env_class("echo")
            EchoEnv2 = AutoEnv.get_env_class("echo-env")
            EchoEnv3 = AutoEnv.get_env_class("echo_env")

            # Should all return the same class
            assert EchoEnv1 is EchoEnv2
            assert EchoEnv2 is EchoEnv3
        except (ValueError, ImportError):
            pytest.skip("echo_env package not properly installed")

    def test_auto_env_get_env_info(self):
        """Test getting environment info."""
        try:
            info = AutoEnv.get_env_info("echo")
            assert info["name"] == "echo_env"
            assert info["env_class"] == "EchoEnv"
            assert info["action_class"] == "EchoAction"
            assert "description" in info
            assert "default_image" in info
            assert "package" in info
            assert info["package"].startswith("openenv-")
        except (ValueError, ImportError):
            pytest.skip("echo_env package not properly installed")

    def test_auto_env_list_environments(self, capsys):
        """Test listing all environments."""
        AutoEnv.list_environments()
        captured = capsys.readouterr()

        assert "Available OpenEnv Environments" in captured.out
        # Should show at least the pattern, even if no envs installed
        assert "Total:" in captured.out

    def test_auto_env_unknown_environment(self):
        """Test error handling for unknown environment."""
        with pytest.raises(ValueError) as exc_info:
            AutoEnv.get_env_class("nonexistent-environment")

        assert "Unknown environment" in str(exc_info.value)

    def test_auto_env_get_env_info_unknown(self):
        """Test getting info for unknown environment."""
        with pytest.raises(ValueError) as exc_info:
            AutoEnv.get_env_info("nonexistent")

        assert "Unknown environment" in str(exc_info.value)


class TestAutoActionIntegration:
    """Test AutoAction integration with package discovery."""

    def setup_method(self):
        """Reset discovery before each test."""
        reset_discovery()

    def test_auto_action_from_name_simple(self):
        """Test getting action class from simple name."""
        try:
            EchoAction = AutoAction.from_name("echo")
            assert EchoAction.__name__ == "EchoAction"
            assert "echo_env" in EchoAction.__module__
        except (ValueError, ImportError):
            pytest.skip("echo_env package not properly installed")

    def test_auto_action_from_name_flexible(self):
        """Test getting action class with different name formats."""
        try:
            # All these should work
            Action1 = AutoAction.from_name("echo")
            Action2 = AutoAction.from_name("echo-env")
            Action3 = AutoAction.from_name("echo_env")

            # Should all return the same class
            assert Action1 is Action2
            assert Action2 is Action3
        except (ValueError, ImportError):
            pytest.skip("echo_env package not properly installed")

    def test_auto_action_from_env(self):
        """Test from_env() alias method."""
        try:
            Action1 = AutoAction.from_name("echo")
            Action2 = AutoAction.from_env("echo")

            # Should return the same class
            assert Action1 is Action2
        except (ValueError, ImportError):
            pytest.skip("echo_env package not properly installed")

    def test_auto_action_coding_env(self):
        """Test with coding_env if installed."""
        try:
            CodeAction = AutoAction.from_name("coding")
            assert CodeAction.__name__ == "CodeAction"
            assert "coding_env" in CodeAction.__module__
        except ValueError:
            pytest.skip("coding_env package not installed")

    def test_auto_action_get_action_info(self):
        """Test getting action info."""
        try:
            info = AutoAction.get_action_info("echo")
            assert info["action_class"] == "EchoAction"
            assert info["env_name"] == "echo_env"
            assert "package" in info
        except (ValueError, ImportError):
            pytest.skip("echo_env package not properly installed")

    def test_auto_action_list_actions(self, capsys):
        """Test listing all action classes."""
        AutoAction.list_actions()
        captured = capsys.readouterr()

        assert "Available Action Classes" in captured.out
        assert "Total:" in captured.out

    def test_auto_action_unknown_environment(self):
        """Test error handling for unknown environment."""
        with pytest.raises(ValueError) as exc_info:
            AutoAction.from_name("nonexistent-environment")

        assert "Unknown environment" in str(exc_info.value)


class TestAutoEnvAutoActionTogether:
    """Test using AutoEnv and AutoAction together."""

    def setup_method(self):
        """Reset discovery before each test."""
        reset_discovery()

    def test_auto_env_and_action_together(self):
        """Test getting both environment and action class."""
        try:
            # Get environment class
            EchoEnv = AutoEnv.get_env_class("echo")
            assert EchoEnv.__name__ == "EchoEnv"

            # Get action class
            EchoAction = AutoAction.from_name("echo")
            assert EchoAction.__name__ == "EchoAction"

            # Verify they're related
            info = AutoEnv.get_env_info("echo")
            assert info["action_class"] == "EchoAction"
        except (ValueError, ImportError):
            pytest.skip("echo_env package not properly installed")

    def test_multiple_environments(self):
        """Test working with multiple environments."""
        try:
            # Try echo
            EchoAction = AutoAction.from_name("echo")
            assert EchoAction is not None

            # Try coding (if installed)
            try:
                CodeAction = AutoAction.from_name("coding")
                assert CodeAction is not None
                # Should be different classes
                assert EchoAction is not CodeAction
            except ValueError:
                # coding_env not installed, that's ok
                pass

        except (ValueError, ImportError):
            pytest.skip("No environment packages properly installed")

    def test_action_creation(self):
        """Test creating action instances."""
        try:
            EchoAction = AutoAction.from_name("echo")

            # Create an action instance
            action = EchoAction(message="Hello, World!")

            # Verify it's the right type
            assert isinstance(action, EchoAction)
            assert hasattr(action, "message")
        except (ValueError, ImportError):
            pytest.skip("echo_env package not properly installed")


class TestDiscoveryPerformance:
    """Test discovery caching and performance."""

    def setup_method(self):
        """Reset discovery before each test."""
        reset_discovery()

    def test_discovery_uses_cache(self):
        """Test that discovery uses cache on subsequent calls."""
        from envs._discovery import get_discovery

        discovery = get_discovery()

        # First call - should discover
        envs1 = discovery.discover(use_cache=False)

        # Second call with cache - should be fast
        envs2 = discovery.discover(use_cache=True)

        # Should return the same data (from cache)
        assert envs1.keys() == envs2.keys()

    def test_cache_invalidation(self):
        """Test that cache can be cleared."""
        from envs._discovery import get_discovery

        discovery = get_discovery()

        # Discover and cache
        discovery.discover()

        # Clear cache
        discovery.clear_cache()

        # Should rediscover
        envs = discovery.discover(use_cache=False)
        assert envs is not None


class TestHubDetection:
    """Test HuggingFace Hub URL detection."""

    def test_hub_url_detection(self):
        """Test that Hub URLs are detected correctly."""
        from envs._discovery import _is_hub_url

        # Hub URLs
        assert _is_hub_url("meta-pytorch/coding-env")
        assert _is_hub_url("org/repo")
        assert _is_hub_url("https://huggingface.co/meta-pytorch/coding-env")

        # Local names
        assert not _is_hub_url("coding")
        assert not _is_hub_url("coding-env")
        assert not _is_hub_url("echo_env")
