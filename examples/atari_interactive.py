#!/usr/bin/env python3
"""
Interactive Atari Example for OpenEnv.
Run this script to see a continuous stream of gameplay in the web interface.
"""

import time
import random
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add repo root to path for 'envs' import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openenv.core.env_client import EnvClient
    from envs.atari_env.models import AtariAction, AtariObservation, AtariState
except ImportError:
    from envs.atari_env import AtariEnv, AtariAction

# Custom client not strictly needed if we just use Generic one, but good for explicit typing
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


async def main():
    print("Connecting to Atari environment...")
    # Port 8011 is where we will run the server
    env = AtariClient(base_url="http://127.0.0.1:8011")

    try:
        await env.connect()
        while True:
            print("Starting new episode...")
            await env.reset()
            
            done = False
            step_count = 0
            
            while not done:
                # 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT usually for Breakout/Pong
                # We'll use random actions to keep it moving
                action_id = random.randint(0, 3) 
                
                await env.step(AtariAction(action_id=action_id))
                
                step_count += 1
                
                # Sleep slightly to make it watchable (20fps approx)
                await asyncio.sleep(0.05)
                
                # Check for reset condition (game over) via side channel or just 
                # run for fixed steps if 'done' isn't reliable in random play
                if step_count > 1000:
                    break
                    
    except KeyboardInterrupt:
        print("\nStopping interaction...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await env.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
