#!/usr/bin/env python3
"""
Interactive Atari Example for OpenEnv.
Run this script to see a continuous stream of gameplay in the web interface.
"""

import time
import random

from openenv.core.env_client import EnvClient
from envs.atari_env.models import AtariAction, AtariObservation, AtariState

# Use the same client class mechanism, but we can define a simple one here
# or import if available. For simplicity we use generic EnvClient.

class AtariClient(EnvClient[AtariAction, AtariObservation, AtariState]):
    def _step_payload(self, action: AtariAction) -> dict:
        return {"action": action.action_id}

    def _parse_result(self, payload: dict):
        # Simplified parsing for the demo
        from openenv.core.client_types import StepResult
        obs = AtariObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False))
        )

    def _parse_state(self, payload: dict) -> AtariState:
        return AtariState(**payload)


def main():
    print("Connecting to Atari environment...")
    # Port 8011 is where we will run the server
    env = AtariClient(base_url="http://127.0.0.1:8011")

    try:
        while True:
            print("Starting new episode...")
            env.reset()
            
            done = False
            step_count = 0
            
            while not done:
                # 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT usually for Breakout/Pong
                # We'll use random actions to keep it moving
                action_id = random.randint(0, 3) 
                
                env.step(AtariAction(action_id=action_id))
                
                step_count += 1
                
                # Sleep slightly to make it watchable (20fps approx)
                time.sleep(0.05)
                
                # Check for reset condition (game over) via side channel or just 
                # run for fixed steps if 'done' isn't reliable in random play
                if step_count > 1000:
                    break
                    
    except KeyboardInterrupt:
        print("\nStopping interaction...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
