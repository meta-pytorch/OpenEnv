#!/usr/bin/env python3
"""
Simple example demonstrating Atari Environment usage.

This example shows how to:
1. Connect to an Atari environment
2. Reset the environment
3. Take random actions
4. Process observations

Usage:
    # First, start the server:
    python -m envs.atari_env.server.app

    # Then run this script:
    python examples/atari_simple.py
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from openenv.core.env_client import EnvClient
    from envs.atari_env.models import AtariAction, AtariObservation, AtariState
except ImportError:
    # Try importing with envs prefix just in case
    try:
        from envs.atari_env import AtariEnv, AtariAction, AtariObservation, AtariState
        # Mapping for compatibility if imported directly
    except ImportError:
        print("Please set PYTHONPATH to include 'src' and 'envs'")
        sys.exit(1)

class AtariClient(EnvClient[AtariAction, AtariObservation, AtariState]):
    def _step_payload(self, action: AtariAction) -> dict:
        return {"action_id": action.action_id}

    def _parse_result(self, payload: dict):
        from openenv.core.client_types import StepResult
        obs_data = payload.get("observation", {})
        obs = AtariObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False))
        )

    def _parse_state(self, payload: dict) -> AtariState:
        return AtariState(**payload)
# import envs
# print(envs.__path__)

def main():
    """Run a simple Atari episode."""
    # Connect to the Atari environment server
    print("Connecting to Atari environment...")
    
    # Simple check for debug flag
    if "--debug" in sys.argv:
        print("Running in DEBUG mode (connecting to http://127.0.0.1:8011)")
        env = AtariClient(base_url="http://127.0.0.1:8011").sync()
    else:
        print("Running in STANDARD mode (using Docker image)")
        # Note: from_docker_image is async, so we need to handle it differently
        # For now, we'll use the debug mode as the primary path
        import asyncio
        async_env = asyncio.run(AtariClient.from_docker_image("ghcr.io/meta-pytorch/openenv-atari-env:latest"))
        env = async_env.sync()
    
   
    try:
        # Reset the environment
        print("\nResetting environment...")
        result = env.reset()
        print(f"Screen shape: {result.observation.screen_shape}")

    
    
        print(f"Legal actions: {result.observation.legal_actions}")
        print(f"Lives: {result.observation.lives}")

        # Run a few steps with random actions
        print("\nTaking random actions...")
        episode_reward = 0
        steps = 0

        for step in range(100):
            # Random action
            action_id = np.random.choice(result.observation.legal_actions)
            action_id = int(action_id)
            
            # Take action
            result = env.step(AtariAction(action_id=action_id))
            import time; time.sleep(0.1)

            episode_reward += result.reward or 0
            steps += 1

            # Print progress
            if step % 10 == 0:
                print(
                    f"Step {step}: reward={result.reward:.2f}, "
                    f"lives={result.observation.lives}, done={result.done}"
                )

            if result.done:
                print(f"\nEpisode finished after {steps} steps!")
                break

        print(f"\nTotal episode reward: {episode_reward:.2f}")

        # Get environment state
        state = env.state()
        print(f"\nEnvironment state:")
        print(f"  Game: {state.game_name}")
        print(f"  Episode: {state.episode_id}")
        print(f"  Steps: {state.step_count}")
        print(f"  Obs type: {state.obs_type}")

    finally:
        # Cleanup
        print("\nClosing environment...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
