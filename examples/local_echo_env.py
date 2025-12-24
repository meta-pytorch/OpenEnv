#!/usr/bin/env python3
"""
Simple test showing how users will use EchoEnv.from_docker_image().

This is the simplest possible usage
"""

import sys
from pathlib import Path

from echo_env import EchoAction, EchoEnv


def main():
    """Test EchoEnv.from_docker_image()."""
    print("=" * 60)
    print("EchoEnv.from_docker_image() Test")
    print("=" * 60)
    print()

    try:
        # This is what users will do - just one line!
        print("Creating client from Docker image...")
        print("  EchoEnv.from_docker_image('echo-env:latest')")
        print()

        client = EchoEnv(base_url="http://localhost:8000")

        print("✓ Client created and container started!\n")

        # Now use it like any other client
        print("Testing the environment:")
        print("-" * 60)

        # Reset
        print("\n1. Reset:")
        result = client.reset()
        print(f"   Message: {result.observation.echoed_message}")
        print(f"   Reward: {result.reward}")
        print(f"   Done: {result.done}")

        # Send some messages
        print("\n2. Send messages:")

        messages = [
            "Hello, World!",
            "Testing echo environment",
            "One more message",
        ]

        for i, msg in enumerate(messages, 1):
            result = client.step(EchoAction(message=msg))
            print(f"   {i}. '{msg}'")
            print(f"      → Echoed: '{result.observation.echoed_message}'")
            print(f"      → Length: {result.observation.message_length}")
            print(f"      → Reward: {result.reward}")

        print("\n" + "-" * 60)
        print("\n✓ All operations successful!")
        print()

        print("Cleaning up...")
        client.close()
        print("✓ Container stopped and removed")
        print()

        print("=" * 60)
        print("Test completed successfully! 🎉")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
