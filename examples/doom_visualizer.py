#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Doom Environment Visualizer.

This script connects to a running Doom environment server and provides
real-time visualization of the game. Use this to see what the agent sees
when the container is running.

Prerequisites:
    1. Build the Docker image:
       docker build -t doom-env:latest -f src/envs/doom_env/server/Dockerfile .

    2. Install visualization library locally:
       pip install opencv-python  # Recommended (supports keyboard controls)
       # or
       pip install matplotlib     # Fallback (no keyboard controls)

Usage:
    # Terminal 1: Start the Docker container
    docker run -p 8000:8000 doom-env:latest

    # Terminal 2: Run this visualizer
    cd examples
    python doom_visualizer.py

    # With higher resolution (recommended for better visibility)
    docker run -p 8000:8000 -e DOOM_SCREEN_RESOLUTION=RES_640X480 doom-env:latest
    python doom_visualizer.py

    # Check what resolution the server is using
    python doom_visualizer.py --resolution-info

    # Connect to a different server
    python doom_visualizer.py --url http://localhost:8000

    # Use matplotlib instead of OpenCV
    python doom_visualizer.py --matplotlib

Controls (OpenCV mode):
    - Arrow keys or A/D: Move left/right
    - Space: Shoot
    - Q or ESC: Quit

Notes:
    - The visualizer displays the game in real-time with health/reward info
    - Docker resolution determines image quality (change with -e DOOM_SCREEN_RESOLUTION)
    - Window automatically scales based on game resolution
    - For best quality: Use RES_640X480 or RES_800X600 on Docker side
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    try:
        import matplotlib.pyplot as plt

        HAS_MPL = True
    except ImportError:
        HAS_MPL = False


def visualize_with_cv2(env):
    """
    Visualize using OpenCV with keyboard controls.

    This function creates an interactive window where you can control the Doom agent
    using keyboard inputs. The window displays the game screen with overlaid info
    (reward, health, episode status).

    Controls:
        - A or Left Arrow: Move left (action_id=1)
        - D or Right Arrow: Move right (action_id=2)
        - Space: Shoot/Attack (action_id=3)
        - Q or ESC: Quit visualizer

    Args:
        env: Connected DoomEnv client instance

    Notes:
        - Window size automatically scales based on game resolution
        - For 160x120 game: Window is ~1024x768 (6x scale)
        - For 640x480 game: Window is ~1024x768 (1.6x scale)
        - Uses INTER_NEAREST interpolation to preserve pixel art aesthetic
    """
    from doom_env import DoomAction

    print("\nControls:")
    print("  Arrow Left / A: Move left")
    print("  Arrow Right / D: Move right")
    print("  Space: Shoot")
    print("  Q or ESC: Quit")
    print("\nStarting visualization...")

    result = env.reset()
    window_name = "Doom Environment - Press Q to quit"

    # Get game resolution
    height, width = result.observation.screen_shape[:2]
    print(f"Game resolution: {width}x{height}")

    # Calculate window size - make it much larger for better visibility
    # Target at least 800 pixels wide
    target_width = 1024
    scale_factor = max(1, target_width // width)
    window_width = width * scale_factor
    window_height = height * scale_factor

    print(f"Window size: {window_width}x{window_height} (scaled {scale_factor}x)")

    # Create window and set size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    # Action mapping for keyboard
    # For basic scenario: [0=noop, 1=left, 2=right, 3=attack]
    running = True
    while running:
        # Get screen
        screen = np.array(result.observation.screen_buffer, dtype=np.uint8).reshape(
            result.observation.screen_shape
        )

        # Convert RGB to BGR for OpenCV
        if len(screen.shape) == 3 and screen.shape[2] == 3:
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        else:
            screen_bgr = screen

        # Resize the image for better display (use interpolation)
        screen_bgr = cv2.resize(
            screen_bgr,
            (window_width, window_height),
            interpolation=cv2.INTER_NEAREST,  # Use INTER_NEAREST to preserve pixel art look
        )

        # Add info overlay
        info_text = [
            f"Reward: {result.reward:.1f}",
            (
                f"Health: {result.observation.game_variables[0]:.0f}"
                if result.observation.game_variables
                else "Health: N/A"
            ),
            f"Done: {result.observation.done}",
            f"Actions: {result.observation.available_actions}",
        ]
        y_pos = 30
        for text in info_text:
            cv2.putText(
                screen_bgr,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y_pos += 25

        cv2.imshow(window_name, screen_bgr)

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        action_id = 0  # Default: no action

        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord("a") or key == 81:  # A or Left arrow
            action_id = 1  # Move left
        elif key == ord("d") or key == 83:  # D or Right arrow
            action_id = 2  # Move right
        elif key == ord(" "):  # Space
            action_id = 3  # Attack
        else:
            pass
            # Random action for demonstration
            # if result.observation.available_actions:
            #     action_id = int(np.random.choice(result.observation.available_actions))

        # Take action
        result = env.step(DoomAction(action_id=action_id))

        if result.observation.done:
            print("\nEpisode finished! Resetting...")
            result = env.reset()
            time.sleep(1)

    cv2.destroyAllWindows()


def visualize_with_matplotlib(env):
    """
    Visualize using Matplotlib (automatic random actions).

    This function creates a matplotlib figure that displays the game in real-time.
    The agent takes random actions automatically. Good for demonstration or if
    OpenCV is not available.

    Controls:
        - Ctrl+C: Stop visualizer
        - Close window: Stop visualizer

    Args:
        env: Connected DoomEnv client instance

    Notes:
        - No keyboard controls (random actions only)
        - Slower than OpenCV
        - Updates every 30ms
        - Shows step count, reward, health in title
    """
    from doom_env import DoomAction

    print("\nStarting visualization (matplotlib mode - no keyboard controls)...")
    print("Taking random actions. Close the window to quit.")

    result = env.reset()
    plt.ion()
    fig = plt.figure(figsize=(10, 7))

    step = 0
    try:
        while True:
            plt.clf()

            # Get and display screen
            screen = np.array(result.observation.screen_buffer, dtype=np.uint8).reshape(
                result.observation.screen_shape
            )

            plt.imshow(screen)
            plt.axis("off")

            # Add info
            info = f"Step: {step} | Reward: {result.reward:.1f}"
            if result.observation.game_variables:
                info += f" | Health: {result.observation.game_variables[0]:.0f}"
            info += f" | Done: {result.observation.done}"
            plt.title(info, fontsize=10, pad=10)

            plt.pause(0.03)

            # Take random action
            if result.observation.available_actions:
                action_id = int(np.random.choice(result.observation.available_actions))
            else:
                action_id = 0

            result = env.step(DoomAction(action_id=action_id))
            step += 1

            if result.observation.done:
                print(f"\nEpisode finished at step {step}! Resetting...")
                result = env.reset()
                step = 0
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        plt.close("all")


def main():
    """
    Main entry point for the Doom environment visualizer.

    This function:
    1. Parses command-line arguments
    2. Checks for required visualization libraries (cv2 or matplotlib)
    3. Connects to the Doom environment server
    4. Launches the appropriate visualization mode
    5. Handles cleanup on exit

    Command-line arguments:
        --url: Server URL (default: http://localhost:8000)
        --matplotlib: Use matplotlib instead of OpenCV
        --resolution-info: Show server resolution and exit

    Examples:
        # Basic usage (connects to localhost:8000)
        python doom_visualizer.py

        # Check server resolution
        python doom_visualizer.py --resolution-info

        # Use matplotlib
        python doom_visualizer.py --matplotlib

        # Connect to remote server
        python doom_visualizer.py --url http://remote-server:8000

    Requirements:
        - Running Doom environment server
        - opencv-python OR matplotlib installed locally
    """
    parser = argparse.ArgumentParser(
        description="Visualize Doom environment in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - connect to local Docker container
  python doom_visualizer.py

  # Check what resolution the server is running
  python doom_visualizer.py --resolution-info

  # Connect to custom URL
  python doom_visualizer.py --url http://localhost:8000

  # Use matplotlib instead of OpenCV
  python doom_visualizer.py --matplotlib

Complete workflow:
  # 1. Rebuild Docker image (if you updated server code)
  docker build -t doom-env:latest -f src/envs/doom_env/server/Dockerfile .

  # 2. Start container with higher resolution (recommended)
  docker run -p 8000:8000 -e DOOM_SCREEN_RESOLUTION=RES_640X480 doom-env:latest

  # 3. Install visualization library locally
  pip install opencv-python

  # 4. Run visualizer
  python doom_visualizer.py

Available resolutions for Docker:
  RES_160X120   - 160×120  (default, very small)
  RES_320X240   - 320×240  (small)
  RES_640X480   - 640×480  (recommended for visualization)
  RES_800X600   - 800×600  (large)
  RES_1024X768  - 1024×768 (very large)

Requirements:
  pip install opencv-python  # Recommended (supports keyboard controls)
  # or
  pip install matplotlib     # Fallback (no keyboard controls)
        """,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="URL of the Doom environment server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Use matplotlib instead of OpenCV for visualization",
    )
    parser.add_argument(
        "--resolution-info",
        action="store_true",
        help="Show current server resolution and exit",
    )

    args = parser.parse_args()

    # Check dependencies
    if not HAS_CV2 and not HAS_MPL:
        print("Error: No visualization library found!")
        print("Install one of the following:")
        print("  pip install opencv-python  (recommended)")
        print("  pip install matplotlib")
        sys.exit(1)

    # Import and create client
    try:
        from doom_env import DoomEnv
    except ImportError as e:
        print(f"Error: Could not import doom_env: {e}")
        print("Make sure you're running from the examples/ directory")
        sys.exit(1)

    print(f"Connecting to Doom environment at {args.url}...")
    try:
        env = DoomEnv(base_url=args.url)
        print("✓ Connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure the server is running:")
        print("  docker run -p 8000:8000 doom-env:latest")
        sys.exit(1)

    # Show resolution info if requested
    if args.resolution_info:
        result = env.reset()
        height, width = result.observation.screen_shape[:2]
        print(f"\nServer Information:")
        print(f"  Screen resolution: {width}x{height}")
        print(f"  Screen shape: {result.observation.screen_shape}")
        print(f"\nTo change resolution, restart Docker with:")
        print(
            f"  docker run -p 8000:8000 -e DOOM_SCREEN_RESOLUTION=RES_640X480 doom-env:latest"
        )
        print(f"\nAvailable resolutions:")
        print(f"  RES_160X120, RES_320X240, RES_640X480, RES_800X600, RES_1024X768")
        env.close()
        return

    try:
        # Choose visualization method
        if args.matplotlib or not HAS_CV2:
            if not HAS_MPL:
                print(
                    "Error: matplotlib not available. Install with: pip install matplotlib"
                )
                sys.exit(1)
            visualize_with_matplotlib(env)
        else:
            visualize_with_cv2(env)
    finally:
        print("\nClosing environment...")
        env.close()
        print("✓ Done!")


if __name__ == "__main__":
    main()
