"""
Simple example script demonstrating the Warehouse Environment.

This script shows basic usage of the warehouse environment with both
random and greedy agents.
"""

import os
import random
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from envs.warehouse_env import WarehouseAction, WarehouseEnv


def random_agent_example():
    """Run a simple random agent in the warehouse."""
    print("=" * 60)
    print("RANDOM AGENT EXAMPLE")
    print("=" * 60)

    # Connect to server (assumes server is running on localhost:8000)
    # Or use from_docker_image() to automatically start a container
    try:
        env = WarehouseEnv(base_url="http://localhost:8000")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("\nPlease start the server first:")
        print("  docker run -p 8000:8000 warehouse-env:latest")
        print("Or run the server locally:")
        print(
            "  python -m uvicorn envs.warehouse_env.server.app:app --host 0.0.0.0 --port 8000"
        )
        return

    # Run one episode
    result = env.reset()
    print(f"\nStarting episode...")

    # Debug: print what we got
    print(f"Grid data: {result.observation.grid}")
    print(f"Robot position: {result.observation.robot_position}")

    if result.observation.grid and len(result.observation.grid) > 0:
        print(
            f"Grid size: {len(result.observation.grid)}x{len(result.observation.grid[0])}"
        )
    else:
        print("Warning: Grid is empty!")

    print(f"Packages to deliver: {result.observation.total_packages}")
    print(f"Max steps: {result.observation.time_remaining}")

    step = 0
    done = False

    while not done and step < 50:  # Limit to 50 steps for demo
        # Random action
        action = WarehouseAction(action_id=random.randint(0, 5))
        result = env.step(action)

        step += 1
        if result.reward != -0.1:  # Only print interesting events
            print(
                f"Step {step}: {action.action_name} -> {result.observation.message} (reward: {result.reward:.1f})"
            )

        done = result.done

    # Print final results
    print(f"\nEpisode finished!")
    print(f"Steps taken: {step}")
    print(
        f"Packages delivered: {result.observation.packages_delivered}/{result.observation.total_packages}"
    )
    print(f"Total reward: {env.state().cum_reward:.2f}")

    env.close()


def greedy_agent_example():
    """Run a simple greedy agent that moves toward targets."""
    print("\n" + "=" * 60)
    print("GREEDY AGENT EXAMPLE")
    print("=" * 60)

    def get_greedy_action(obs):
        """Simple greedy policy: move toward nearest target."""
        robot_x, robot_y = obs.robot_position

        # Determine target location
        if obs.robot_carrying is None:
            # Not carrying: move toward nearest waiting package
            for pkg in obs.packages:
                if pkg["status"] == "waiting":
                    target_x, target_y = pkg["pickup_location"]
                    break
            else:
                # No packages waiting, try to pick up
                return 4
        else:
            # Carrying: move toward dropoff zone
            pkg = next((p for p in obs.packages if p["id"] == obs.robot_carrying), None)
            if pkg:
                target_x, target_y = pkg["dropoff_location"]
            else:
                return 4  # Try action

        # Simple pathfinding: move closer on one axis at a time
        if robot_x < target_x:
            return 3  # RIGHT
        elif robot_x > target_x:
            return 2  # LEFT
        elif robot_y < target_y:
            return 1  # DOWN
        elif robot_y > target_y:
            return 0  # UP
        else:
            # At target location
            return 4 if obs.robot_carrying is None else 5

    try:
        env = WarehouseEnv(base_url="http://localhost:8000")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    # Run 3 episodes
    for episode in range(3):
        result = env.reset()
        print(f"\nEpisode {episode + 1}")
        print(f"Packages: {result.observation.total_packages}")

        done = False
        steps = 0

        while not done and steps < 200:
            action_id = get_greedy_action(result.observation)
            action = WarehouseAction(action_id=action_id)
            result = env.step(action)
            steps += 1

            # Print delivery events
            if "delivered" in result.observation.message.lower():
                print(f"  Step {steps}: {result.observation.message}")

            done = result.done

        state = env.state()
        print(
            f"  Result: {state.packages_delivered}/{state.total_packages} delivered, "
            f"reward: {state.cum_reward:.2f}, steps: {steps}"
        )

    env.close()


def visualization_example():
    """Demonstrate ASCII visualization."""
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLE")
    print("=" * 60)

    try:
        env = WarehouseEnv(base_url="http://localhost:8000")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    # Reset and show initial state
    result = env.reset()
    print("\nInitial warehouse state:")
    print(env.render_ascii())

    # Take a few actions and show updates
    actions = [3, 3, 1, 1, 4]  # RIGHT, RIGHT, DOWN, DOWN, PICKUP
    for i, action_id in enumerate(actions):
        action = WarehouseAction(action_id=action_id)
        result = env.step(action)
        print(f"\nAfter action {i+1} ({action.action_name}):")
        print(env.render_ascii())

        if result.done:
            break

    env.close()


if __name__ == "__main__":
    print("Warehouse Environment - Example Script")
    print("=" * 60)
    print("\nThis script demonstrates the warehouse environment.")
    print("Make sure the server is running on http://localhost:8000")
    print("\nTo start the server:")
    print("  docker run -p 8000:8000 warehouse-env:latest")
    print("\n" + "=" * 60)

    # Run examples
    try:
        random_agent_example()
        greedy_agent_example()
        visualization_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
