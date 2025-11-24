#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate GIFs of different Doom scenarios for documentation.

This script creates animated GIFs showcasing various ViZDoom scenarios
for use in the README and documentation.
"""

import argparse
import os
from pathlib import Path

import numpy as np


def generate_scenario_gif(
    scenario: str,
    output_path: str,
    num_frames: int = 100,
    fps: int = 15,
    screen_resolution: str = "RES_320X240",
):
    """
    Generate a GIF of a Doom scenario.

    Args:
        scenario: Name of the scenario to run
        output_path: Path to save the GIF
        num_frames: Number of frames to capture
        fps: Frames per second for the GIF
        screen_resolution: Screen resolution to use
    """
    try:
        from server.doom_env_environment import DoomEnvironment
        from models import DoomAction
        import imageio
    except ImportError as e:
        print(f"Error: Missing dependencies. Install with:")
        print(f"  pip install imageio vizdoom")
        print(f"Error details: {e}")
        return False

    print(f"Generating GIF for scenario: {scenario}")
    print(f"  Frames: {num_frames}, FPS: {fps}, Resolution: {screen_resolution}")

    try:
        # Create environment
        env = DoomEnvironment(
            scenario=scenario,
            screen_resolution=screen_resolution,
            screen_format="RGB24",
            window_visible=False,
            use_discrete_actions=True,
        )

        # Reset and collect frames
        obs = env.reset()
        frames = []

        print(f"  Collecting frames...")
        for i in range(num_frames):
            # Get screen as numpy array
            screen = np.array(obs.screen_buffer, dtype=np.uint8).reshape(
                obs.screen_shape
            )
            frames.append(screen)

            # Take a random action
            if obs.available_actions:
                # Bias towards movement and shooting actions
                available = obs.available_actions
                if len(available) > 3:
                    # Prefer actions 1-7 (movement/shooting) over 0 (no-op)
                    action_id = np.random.choice(available[1:])
                else:
                    action_id = np.random.choice(available)
            else:
                action_id = 0

            obs = env.step(DoomAction(action_id=action_id))

            # Reset if episode ends
            if obs.done:
                print(f"    Episode ended at frame {i}, resetting...")
                obs = env.reset()

            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{num_frames} frames")

        env.close()

        # Save as GIF
        print(f"  Saving GIF to: {output_path}")
        imageio.mimsave(
            output_path,
            frames,
            fps=fps,
            loop=0,  # Loop forever
        )

        print(f"✓ Successfully created GIF: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"✗ Error generating GIF for {scenario}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate GIFs of Doom scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate GIFs for all default scenarios
  python generate_gifs.py

  # Generate GIF for a specific scenario
  python generate_gifs.py --scenario basic

  # Generate with custom settings
  python generate_gifs.py --scenario deadly_corridor --frames 150 --fps 20

  # Generate for multiple scenarios
  python generate_gifs.py --scenario basic deadly_corridor defend_the_center

Available scenarios:
  - basic: Simple scenario with basic movement and shooting
  - deadly_corridor: Navigate a corridor avoiding/killing monsters
  - defend_the_center: Defend the center position
  - defend_the_line: Defend a line against monsters
  - health_gathering: Collect health packs to survive
  - my_way_home: Navigate to a specific location
  - predict_position: Predict object positions
  - take_cover: Learn to take cover from enemy fire
        """,
    )

    parser.add_argument(
        "--scenario",
        "-s",
        nargs="+",
        default=["basic", "deadly_corridor", "defend_the_center", "health_gathering"],
        help="Scenario(s) to generate GIFs for",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="assets",
        help="Output directory for GIFs (default: assets)",
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=int,
        default=100,
        help="Number of frames to capture (default: 100)",
    )
    parser.add_argument(
        "--fps", type=int, default=15, help="Frames per second (default: 15)"
    )
    parser.add_argument(
        "--resolution",
        "-r",
        default="RES_320X240",
        choices=[
            "RES_160X120",
            "RES_320X240",
            "RES_640X480",
            "RES_800X600",
            "RES_1024X768",
        ],
        help="Screen resolution (default: RES_320X240)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"Doom Scenario GIF Generator")
    print(f"{'='*60}\n")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Scenarios: {', '.join(args.scenario)}")
    print(f"Settings: {args.frames} frames @ {args.fps} fps, {args.resolution}")
    print()

    # Generate GIFs for each scenario
    results = {}
    for scenario in args.scenario:
        output_file = output_dir / f"{scenario}.gif"

        success = generate_scenario_gif(
            scenario=scenario,
            output_path=str(output_file),
            num_frames=args.frames,
            fps=args.fps,
            screen_resolution=args.resolution,
        )

        results[scenario] = success
        print()

    # Print summary
    print(f"{'='*60}")
    print(f"Summary")
    print(f"{'='*60}\n")

    successful = [s for s, success in results.items() if success]
    failed = [s for s, success in results.items() if not success]

    print(f"✓ Successful: {len(successful)}/{len(results)}")
    for scenario in successful:
        print(f"  - {scenario}.gif")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for scenario in failed:
            print(f"  - {scenario}")

    print(f"\nGIFs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
