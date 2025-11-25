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

Usage:
    # Terminal 1: Start the Docker container
    docker run -p 8000:8000 doom-env:latest

    # Terminal 2: Run this visualizer
    python doom_visualizer.py

    # Or connect to a different server
    python doom_visualizer.py --url http://localhost:8000

Controls:
    - Arrow keys or WASD: Control the agent
    - Space: Shoot
    - Q or ESC: Quit
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
    """Visualize using OpenCV (supports keyboard controls)."""
    from doom_env import DoomAction

    print("\nControls:")
    print("  Arrow Left / A: Move left")
    print("  Arrow Right / D: Move right")
    print("  Space: Shoot")
    print("  Q or ESC: Quit")
    print("\nStarting visualization...")

    result = env.reset()
    window_name = "Doom Environment - Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

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
            running = False
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
    """Visualize using Matplotlib (no keyboard controls)."""
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize Doom environment in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to local Docker container
  python doom_visualizer.py

  # Connect to custom URL
  python doom_visualizer.py --url http://localhost:8000

  # Use matplotlib instead of OpenCV
  python doom_visualizer.py --matplotlib

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
