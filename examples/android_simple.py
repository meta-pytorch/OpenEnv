"""Simple Android Environment Example.

This example demonstrates basic interaction with the Android environment using
gestures, text input, and button presses.

Prerequisites:
- Docker with KVM device access (Linux only for hardware acceleration)
- Android environment Docker image built:
    docker build -t android-env:latest -f src/envs/android_env/server/Dockerfile .
- Task definition file (see src/envs/android_env/examples/tasks/ for examples)

Usage:
    # Set environment variables
    export ANDROID_TASK_PATH=/workspace/tasks/calculator_basic.textproto
    export ANDROID_AVD_NAME=default_pixel_6

    # Run example
    python examples/android_simple.py

Note: Without KVM (macOS/Windows), the emulator will be very slow.
Use a Linux VM or cloud instance for acceptable performance.
"""

import os
import sys
import time
import base64
from io import BytesIO
from PIL import Image

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from envs.android_env import AndroidEnv, AndroidAction


def decode_observation(screen_image: str):
    """Decode base64 observation to PIL Image."""
    if screen_image.startswith("shm://"):
        print("⚠️  Shared memory observations require client on same machine")
        return None

    image_bytes = base64.b64decode(screen_image)
    return Image.open(BytesIO(image_bytes))


def main():
    """Run simple Android environment example."""

    # Configuration from environment variables
    task_path = os.getenv("ANDROID_TASK_PATH", "/workspace/tasks/calculator_basic.textproto")
    avd_name = os.getenv("ANDROID_AVD_NAME", "default_pixel_6")

    print("=" * 60)
    print("Android Environment - Simple Example")
    print("=" * 60)
    print(f"Task: {task_path}")
    print(f"AVD: {avd_name}")
    print()

    # Connect to Android environment
    # Option 1: Use from_docker_image (recommended)
    print("🚀 Starting Android environment from Docker...")
    client = AndroidEnv.from_docker_image(
        "android-env:latest",
        environment={
            "ANDROID_AVD_NAME": avd_name,
            "ANDROID_TASK_PATH": task_path,
            "ANDROID_RUN_HEADLESS": "true",
            "ANDROID_IMAGE_FORMAT": "JPEG",
            "ANDROID_IMAGE_QUALITY": "85",
        },
        volumes={
            os.path.join(os.path.dirname(__file__), "..", "src", "envs", "android_env", "examples", "tasks"):
                "/workspace/tasks"
        },
        device_requests=[
            {
                "PathOnHost": "/dev/kvm",
                "PathInContainer": "/dev/kvm",
                "CgroupPermissions": "rwm"
            }
        ] if sys.platform == "linux" else None,  # KVM only on Linux
        timeout=120  # Emulator boot can take 60+ seconds
    )

    # Option 2: Connect to existing server
    # client = AndroidEnv(base_url="http://localhost:8000", timeout=120)

    try:
        # Reset environment
        print("\n📱 Resetting environment (this may take 30-60 seconds on first boot)...")
        result = client.reset()
        obs = result.observation

        print(f"✅ Environment ready!")
        print(f"   Screen: {obs.screen_width}x{obs.screen_height}")
        print(f"   Orientation: {obs.orientation}°")
        print(f"   Image size: {len(obs.screen_image)} bytes")
        print()

        # Decode and optionally save first observation
        img = decode_observation(obs.screen_image)
        if img:
            img.save("/tmp/android_initial_screen.jpg")
            print("💾 Saved initial screen to /tmp/android_initial_screen.jpg")
        print()

        # Example 1: Tap at center
        print("Example 1: Tap at center of screen")
        action = AndroidAction(tool_name="tap", parameters={"x": 0.5, "y": 0.5})
        result = client.step(action)
        print(f"   Result: reward={result.reward}, done={result.done}")
        time.sleep(1)

        # Example 2: Swipe down (scroll)
        print("\nExample 2: Swipe down (scroll)")
        action = AndroidAction(
            tool_name="swipe",
            parameters={"x1": 0.5, "y1": 0.7, "x2": 0.5, "y2": 0.3}
        )
        result = client.step(action)
        print(f"   Result: reward={result.reward}, done={result.done}")
        time.sleep(1)

        # Example 3: Long press
        print("\nExample 3: Long press at (0.3, 0.3)")
        action = AndroidAction(
            tool_name="long_press",
            parameters={"x": 0.3, "y": 0.3, "duration_ms": 1000}
        )
        result = client.step(action)
        print(f"   Result: reward={result.reward}, done={result.done}")
        time.sleep(1)

        # Example 4: Type text (if supported by task)
        print("\nExample 4: Type text")
        action = AndroidAction(tool_name="type_text", parameters={"text": "Hello Android"})
        result = client.step(action)
        print(f"   Result: reward={result.reward}, done={result.done}")
        time.sleep(1)

        # Example 5: Press HOME button
        print("\nExample 5: Press HOME button")
        action = AndroidAction(tool_name="press_button", parameters={"button": "HOME"})
        result = client.step(action)
        print(f"   Result: reward={result.reward}, done={result.done}")
        time.sleep(1)

        # Example 6: Double tap
        print("\nExample 6: Double tap at (0.7, 0.7)")
        action = AndroidAction(tool_name="double_tap", parameters={"x": 0.7, "y": 0.7})
        result = client.step(action)
        print(f"   Result: reward={result.reward}, done={result.done}")
        time.sleep(1)

        # Example 7: Scroll using helper actions
        print("\nExample 7: Scroll down using scroll_down action")
        action = AndroidAction(tool_name="scroll_down", parameters={"distance": 0.5})
        result = client.step(action)
        print(f"   Result: reward={result.reward}, done={result.done}")
        time.sleep(1)

        print("\nExample 8: Scroll up using scroll_up action")
        action = AndroidAction(tool_name="scroll_up", parameters={"distance": 0.5})
        result = client.step(action)
        print(f"   Result: reward={result.reward}, done={result.done}")
        time.sleep(1)

        # Get final observation
        print("\n📊 Final observation:")
        final_obs = result.observation
        print(f"   Screen: {final_obs.screen_width}x{final_obs.screen_height}")
        print(f"   Done: {final_obs.done}")
        print(f"   Total reward: {result.reward}")

        # Decode and save final observation
        img = decode_observation(final_obs.screen_image)
        if img:
            img.save("/tmp/android_final_screen.jpg")
            print("💾 Saved final screen to /tmp/android_final_screen.jpg")

        print("\n✅ Example completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        client.close()
        print("Done!")


def run_simple_loop():
    """Run a simple random action loop (alternative example)."""
    import random

    client = AndroidEnv.from_docker_image(
        "android-env:latest",
        environment={
            "ANDROID_AVD_NAME": "default_pixel_6",
            "ANDROID_TASK_PATH": "/workspace/tasks/calculator_basic.textproto",
        }
    )

    try:
        result = client.reset()
        print(f"Initial state: {result.observation.screen_width}x{result.observation.screen_height}")

        # Random action loop
        for step in range(10):
            # Random tap
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)

            action = AndroidAction(tool_name="tap", parameters={"x": x, "y": y})
            result = client.step(action)

            print(f"Step {step}: tap({x:.2f}, {y:.2f}) -> reward={result.reward}, done={result.done}")

            if result.done:
                print("Episode ended!")
                break

            time.sleep(0.5)
    finally:
        client.close()


if __name__ == "__main__":
    # Run main example
    main()

    # Uncomment to run simple loop instead:
    # run_simple_loop()
