#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test script for OpenApp Environment.

This script tests the basic functionality of the OpenApp environment
to ensure it follows OpenEnv standards.

Usage:
    # From OpenEnv root directory
    python3 envs/openapp_env/test_openapp_env.py

    # Or from openapp_env directory
    cd envs/openapp_env
    python3 test_openapp_env.py
"""

import sys
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openapp_env.models import OpenAppAction, OpenAppObservation
from openapp_env.server.openapp_environment import OpenAppEnvironment


def test_models():
    """Test that models are properly defined."""
    print("Testing models...")

    # Test creating an action
    action = OpenAppAction(action_type="noop")
    assert action.action_type == "noop"

    # Test click action
    click_action = OpenAppAction(action_type="click", bid="test-btn")
    assert click_action.bid == "test-btn"

    # Test fill action
    fill_action = OpenAppAction(action_type="fill", bid="input", text="Hello")
    assert fill_action.text == "Hello"

    print("✓ Models test passed")


def test_environment_basic():
    """Test basic environment functionality."""
    print("\nTesting environment...")

    try:
        # Create environment (note: this will check if OpenApps is installed as a package)
        env = OpenAppEnvironment(
            max_steps=10,
        )

        # Test that environment has required methods
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "state")
        assert hasattr(env, "close")

        print("✓ Environment structure test passed")

    except (ValueError, ImportError) as e:
        # Expected if OpenApps is not installed as a package
        if "OpenApps not found" in str(e) or "open_apps" in str(e):
            print(
                "✓ Environment structure test passed (OpenApps not installed, expected)"
            )
        else:
            raise


def test_client_server_contract():
    """Test that client and server follow the contract."""
    print("\nTesting client-server contract...")

    # Test that action can be serialized to dict
    action = OpenAppAction(
        action_type="click", bid="test-123", metadata={"test": "value"}
    )

    # Simulate what client._step_payload would do
    payload = {
        "action_type": action.action_type,
        "bid": action.bid,
        "metadata": action.metadata,
    }

    assert payload["action_type"] == "click"
    assert payload["bid"] == "test-123"

    # Test observation construction
    obs = OpenAppObservation(
        html="<html></html>",
        url="http://localhost:5001",
        open_pages_urls=["http://localhost:5001"],
        done=False,
        reward=0.0,
    )

    assert obs.url == "http://localhost:5001"
    assert obs.done is False

    print("✓ Client-server contract test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenApp Environment - Structure Tests")
    print("=" * 60)

    try:
        test_models()
        test_environment_basic()
        test_client_server_contract()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nNote: Full integration tests require:")
        print(
            "1. OpenApps installed: pip install git+https://github.com/facebookresearch/OpenApps.git"
        )
        print("2. Playwright browsers installed: playwright install chromium")
        print("3. BrowserGym dependencies installed")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
