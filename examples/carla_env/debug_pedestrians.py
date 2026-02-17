#!/usr/bin/env python3
"""
Debug script to check pedestrian spawn locations.
"""
import sys
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Add envs directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "envs"))

from carla_env import CarlaEnv, CarlaAction


def debug_pedestrian_spawn(base_url: str, scenario_name: str):
    """Debug where pedestrians are spawning."""

    print("="*70)
    print(f"Debugging Pedestrian Spawn: {scenario_name}")
    print("="*70)

    env = CarlaEnv(base_url=base_url).sync()

    with env:
        # Reset to scenario
        print("\n1. Resetting scenario...")
        result = env.reset(scenario_name=scenario_name)
        obs = result.observation

        print(f"   ✓ Reset complete")
        print(f"   Speed: {obs.speed_kmh:.1f} km/h")
        print(f"   Location: {obs.location}")
        print(f"\n2. Scene description:")
        print(f"   {obs.scene_description}\n")

        # Check nearby actors
        print("3. Nearby actors:")
        if obs.nearby_actors:
            for i, actor in enumerate(obs.nearby_actors, 1):
                print(f"   Actor {i}:")
                print(f"     Type: {actor.get('type')}")
                print(f"     Distance: {actor.get('distance'):.1f}m")
                print(f"     Position: {actor.get('position')}")
        else:
            print("   ❌ NO ACTORS FOUND!")

        # Capture image
        print("\n4. Capturing camera image...")
        result = env.step(CarlaAction(action_type="capture_image"))
        obs = result.observation

        if obs.camera_image:
            image_data = base64.b64decode(obs.camera_image)
            output_file = Path(f"debug_{scenario_name}.jpg")
            output_file.write_bytes(image_data)
            print(f"   ✓ Image saved: {output_file}")
            print(f"   Size: {len(image_data)} bytes")
        else:
            print("   ❌ No image captured")

        # Take a few steps and check again
        print("\n5. Taking 5 steps forward...")
        for step in range(5):
            result = env.step(CarlaAction(action_type="observe"))
            obs = result.observation
            print(f"   Step {step+1}: speed={obs.speed_kmh:.1f} km/h, actors={len(obs.nearby_actors)}, collision={obs.collision_detected}")

        # Final check
        print("\n6. Final state:")
        print(f"   Nearby actors: {len(obs.nearby_actors)}")
        if obs.nearby_actors:
            for actor in obs.nearby_actors[:3]:  # Show first 3
                print(f"     - {actor.get('type')}: {actor.get('distance'):.1f}m {actor.get('position')}")

        print("\n" + "="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--scenario", default="trolley_saves")

    args = parser.parse_args()

    debug_pedestrian_spawn(args.base_url, args.scenario)
