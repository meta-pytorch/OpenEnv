#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive AutoEnv and AutoAction Example
=============================================

This example demonstrates how to use the AutoEnv and AutoAction classes
to automatically select and use environments without manual imports.

The AutoEnv/AutoAction API follows the HuggingFace pattern, making it easy
to work with different environments using a consistent interface.

Run this example with:
    python examples/auto_env_example.py

Or test a specific environment:
    python examples/auto_env_example.py --env coding
"""

import sys
import argparse
from pathlib import Path

from envs import AutoEnv, AutoAction


def example_basic_usage():
    """Example 1: Basic usage with AutoEnv and AutoAction"""
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    print()

    # Instead of:
    # from envs.coding_env import CodingEnv, CodeAction
    # client = CodingEnv.from_docker_image("coding-env:latest")

    # You can now do:
    print("Creating environment using AutoEnv...")
    client = AutoEnv.from_name("coding-env")
    print("✓ Environment created!")
    print()

    # Get the Action class automatically
    print("Getting Action class using AutoAction...")
    CodeAction = AutoAction.from_name("coding-env")
    print(f"✓ Got Action class: {CodeAction.__name__}")
    print()

    # Use them together
    print("Testing the environment:")
    result = client.reset()
    print(f"  Reset: exit_code={result.observation.exit_code}")

    action = CodeAction(code="print('Hello from AutoEnv!')")
    step_result = client.step(action)
    print(f"  Step result: {step_result.observation.stdout.strip()}")

    client.close()
    print("✓ Environment closed")
    print()


def example_alternative_syntax():
    """Example 2: Alternative syntax using environment key"""
    print("=" * 70)
    print("Example 2: Alternative Syntax")
    print("=" * 70)
    print()

    # You can also use just the environment key
    print("Getting Action class by environment name...")
    CodeAction = AutoAction.from_name("coding")
    print(f"✓ Got Action class: {CodeAction.__name__}")
    print()

    # Create instance
    action = CodeAction(code="x = 5 + 3\nprint(f'Result: {x}')")
    print(f"Created action: {action}")
    print()


def example_list_environments():
    """Example 3: List all available environments"""
    print("=" * 70)
    print("Example 3: List Available Environments")
    print("=" * 70)
    print()

    # List all available environments
    AutoEnv.list_environments()
    print()


def example_list_actions():
    """Example 4: List all available action classes"""
    print("=" * 70)
    print("Example 4: List Available Action Classes")
    print("=" * 70)
    print()

    # List all available action classes
    AutoAction.list_actions()
    print()


def example_environment_info():
    """Example 5: Get detailed environment information"""
    print("=" * 70)
    print("Example 5: Environment Information")
    print("=" * 70)
    print()

    # Get detailed info about a specific environment
    env_name = "coding"
    print(f"Information about '{env_name}' environment:")
    print("-" * 70)

    info = AutoEnv.get_env_info(env_name)
    print(f"  Description: {info['description']}")
    print(f"  Docker Image: {info['default_image']}")
    print(f"  Environment Class: {info['env_class']}")
    print(f"  Action Class: {info['action_class']}")
    print(f"  Observation Class: {info['observation_class']}")
    print(f"  Module: {info['module']}")
    print(f"  Version: {info['version']}")
    print(f"  Spec Version: {info['spec_version']}")
    print()


def example_error_handling():
    """Example 6: Error handling with helpful messages"""
    print("=" * 70)
    print("Example 6: Error Handling")
    print("=" * 70)
    print()

    # Try an unknown environment
    print("Trying unknown environment 'nonexistent'...")
    try:
        env = AutoEnv.from_name("nonexistent-env")
    except ValueError as e:
        print(f"✓ Got expected error: {e}")
    print()

    # Try a typo - should suggest similar names
    print("Trying typo 'cooding' (should suggest 'coding')...")
    try:
        env = AutoEnv.from_name("cooding-env")
    except ValueError as e:
        print(f"✓ Got helpful suggestion: {e}")
    print()

    # Try deprecated julia environment
    print("Trying deprecated 'julia' environment...")
    try:
        env = AutoEnv.from_name("julia-env")
    except ValueError as e:
        print(f"✓ Got deprecation notice: {e}")
    print()


def example_special_requirements():
    """Example 7: Environments with special requirements"""
    print("=" * 70)
    print("Example 7: Special Requirements")
    print("=" * 70)
    print()

    # DIPG environment requires dataset path
    print("DIPG environment requires DIPG_DATASET_PATH:")
    print()
    print("  # This would show a warning:")
    print("  # env = AutoEnv.from_name('dipg-env')")
    print()
    print("  # Correct usage:")
    print("  env = AutoEnv.from_name(")
    print("      'dipg-env',")
    print("      env_vars={'DIPG_DATASET_PATH': '/data/dipg'}")
    print("  )")
    print()

    # FinRL environment has optional config
    print("FinRL environment accepts optional config:")
    print()
    print("  env = AutoEnv.from_name(")
    print("      'finrl-env',")
    print("      env_vars={'FINRL_CONFIG_PATH': '/config.json'}")
    print("  )")
    print()


def test_specific_environment(env_name: str):
    """Test a specific environment by name"""
    print("=" * 70)
    print(f"Testing {env_name} Environment")
    print("=" * 70)
    print()

    try:
        # Get environment info
        info = AutoEnv.get_env_info(env_name)
        image = info["default_image"]

        print(f"Creating {env_name} environment...")
        print(f"  Docker image: {image}")
        print()

        # Create environment with extended timeout for slow containers
        # Use the simplified name format
        env_image_name = f"{env_name}-env" if not env_name.endswith("-env") else env_name
        env = AutoEnv.from_name(env_image_name, wait_timeout=60.0)
        print("✓ Environment created!")

        # Get action class
        ActionClass = AutoAction.from_name(env_name)
        print(f"✓ Action class: {ActionClass.__name__}")
        print()

        # Test reset
        print("Testing reset()...")
        result = env.reset()
        print(f"✓ Reset successful")
        print()

        # Get state
        state = env.state()
        print(f"State: episode_id={state.episode_id}, step_count={state.step_count}")
        print()

        # Close
        env.close()
        print("✓ Environment closed")
        print()

        print("=" * 70)
        print(f"✓ {env_name} environment test passed!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Error testing {env_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to run examples"""
    parser = argparse.ArgumentParser(
        description="AutoEnv and AutoAction Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Test a specific environment (e.g., coding, echo, git)",
    )
    parser.add_argument(
        "--all-examples",
        action="store_true",
        help="Run all examples (without Docker)",
    )

    args = parser.parse_args()

    if args.env:
        # Test specific environment
        success = test_specific_environment(args.env)
        sys.exit(0 if success else 1)

    elif args.all_examples:
        # Run all examples (no Docker needed)
        example_basic_usage()  # This requires Docker
        # Skip Docker examples, run info-only examples
        example_alternative_syntax()
        example_list_environments()
        example_list_actions()
        example_environment_info()
        example_error_handling()
        example_special_requirements()

    else:
        # Show usage info and examples that don't need Docker
        print("AutoEnv and AutoAction Examples")
        print("=" * 70)
        print()
        print("This demonstrates the HuggingFace-style API for OpenEnv.")
        print()
        print("Usage:")
        print("  python examples/auto_env_example.py --all-examples")
        print("  python examples/auto_env_example.py --env coding")
        print()
        print("Running info examples (no Docker required)...")
        print()

        example_list_environments()
        example_list_actions()
        example_environment_info()
        example_error_handling()
        example_special_requirements()

        print()
        print("To test with actual Docker environments:")
        print("  python examples/auto_env_example.py --env coding")
        print()


if __name__ == "__main__":
    main()
