import sys
import os

# This adds the current directory (OpenEnv) to Python's search path,
# so it can find the 'grid_world' folder.
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_dir)

# Now we can import from the grid_world package.
from grid_world.client import GridWorldEnv

from grid_world.models import MoveAction

print("Attempting to start GridWorld environment...")

try:
    # This will start the "grid_world:latest" Docker image
    client = GridWorldEnv.from_docker_image("grid_world:latest")

    print("--- Starting Game ---")
    result = client.reset()
    print(f"Initial Observation: {result.observation}")

    # Try a few moves
    print("Taking action: DOWN")
    result = client.step(action=MoveAction.DOWN)
    print(f"  -> Obs: {result.observation}")
    print(f"  -> Reward: {result.reward}")

    print("Taking action: RIGHT")
    result = client.step(action=MoveAction.RIGHT)
    print(f"  -> Obs: {result.observation}")
    print(f"  -> Reward: {result.reward}")

    print("Taking action: UP (trying to hit wall)")
    result = client.step(action=MoveAction.UP) # Should go back to [0, 1]
    result = client.step(action=MoveAction.UP) # Should hit wall
    print(f"  -> Obs: {result.observation}")
    print(f"  -> Reward: {result.reward}") # Should be -0.5

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # This stops and removes the container
    print("--- Cleaning up ---")
    if 'client' in locals() and client:
        client.close()
    print("Test complete.")