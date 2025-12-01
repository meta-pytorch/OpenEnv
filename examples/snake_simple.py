#!/usr/bin/env python3
"""
Interactive Snake Game Player.

This script lets you play the snake game with keyboard controls
or run with an automated agent.

Usage:
    # Option 1: Use Docker
    python examples/snake_rl/play_snake.py --mode docker --play-mode auto

    # Option 2: Use local server
    # Terminal 1: cd src/envs/snake_env && uv run --project . server
    # Terminal 2: python examples/snake_rl/play_snake.py --mode local --play-mode auto
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from envs.snake_env import SnakeAction, SnakeEnv


# Cell type constants (from marlenv.envs.snake_env.Cell)
CELL_EMPTY = 0
CELL_WALL = 1
CELL_FRUIT = 2
CELL_HEAD = 3  # Snake head
CELL_BODY = 4  # Snake body segments
CELL_TAIL = 5  # Snake tail


class SnakeGamePlayer:
    """Interactive snake game player with visualization."""

    def __init__(self, client):
        """Initialize the game player."""
        self.client = client
        self.result = None
        self.running = True
        self.paused = False
        self.current_action = 0  # Default: no-op

        # For automated play
        self.auto_play = False
        self.play_speed = 200  # milliseconds between moves

        # Statistics
        self.episode_count = 0
        self.best_score = 0
        self.best_fruits = 0

    def reset_game(self):
        """Reset the game to initial state."""
        self.result = self.client.reset()
        self.episode_count += 1
        print(f"\n=== Episode {self.episode_count} Started ===")
        return self.result

    def step_game(self, action):
        """Take a step in the game."""
        self.result = self.client.step(SnakeAction(action=action))

        # Update best scores
        if self.result.observation.episode_score > self.best_score:
            self.best_score = self.result.observation.episode_score
        if self.result.observation.episode_fruits > self.best_fruits:
            self.best_fruits = self.result.observation.episode_fruits

        # Print events
        if self.result.reward > 0:
            print(
                f"  🍎 Fruit collected! Score: {self.result.observation.episode_score:.2f}"
            )
        elif self.result.done:
            print(
                f"  ☠️  Game Over! Final score: {self.result.observation.episode_score:.2f}"
            )
            print(
                f"     Steps: {self.result.observation.episode_steps}, "
                f"Fruits: {self.result.observation.episode_fruits}"
            )

        return self.result

    def create_grid_colors(self, grid):
        """Convert grid to colored array for visualization."""
        grid_array = np.array(grid)
        height, width = grid_array.shape
        colored_grid = np.zeros((height, width, 3))

        for i in range(height):
            for j in range(width):
                cell = grid_array[i, j]
                cell_type = cell % 10

                if cell_type == CELL_EMPTY:
                    colored_grid[i, j] = [0.95, 0.95, 0.95]  # Light gray
                elif cell_type == CELL_WALL:
                    colored_grid[i, j] = [0.2, 0.2, 0.2]  # Dark gray
                elif cell_type == CELL_FRUIT:
                    colored_grid[i, j] = [1.0, 0.0, 0.0]  # Red
                elif cell_type == CELL_BODY:
                    colored_grid[i, j] = [0.0, 0.8, 0.0]  # Green
                elif cell_type == CELL_HEAD:
                    colored_grid[i, j] = [0.0, 1.0, 0.0]  # Bright green
                elif cell_type == CELL_TAIL:
                    colored_grid[i, j] = [0.0, 0.6, 0.0]  # Dark green

        return colored_grid

    def play_automated(self, max_steps=500, policy="random"):
        """
        Play the game with an automated agent.

        Args:
            max_steps: Maximum steps per episode
            policy: 'random' or 'simple' (simple tries to move towards fruit)
        """
        print("\n" + "=" * 70)
        print(f"  Automated Play - Policy: {policy}")
        print("=" * 70)

        self.reset_game()
        steps = 0

        # Setup visualization
        fig, (ax_game, ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        plt.ion()  # Interactive mode

        rewards_history = []
        scores_history = []

        while not self.result.done and steps < max_steps:
            # Choose action based on policy
            if policy == "random":
                import random

                action = random.randint(0, 2)
            elif policy == "simple":
                # Simple heuristic: try to stay alive
                action = 0  # Usually go straight
                if steps % 20 == 0:  # Occasional turns
                    import random

                    action = random.randint(0, 2)
            else:
                action = 0

            # Take step
            self.step_game(action)
            steps += 1

            # Update histories
            rewards_history.append(self.result.reward)
            scores_history.append(self.result.observation.episode_score)

            # Update visualization every few steps
            if steps % 5 == 0:
                # Clear axes
                ax_game.clear()
                ax_stats.clear()

                # Game grid
                colored_grid = self.create_grid_colors(self.result.observation.grid)
                ax_game.imshow(colored_grid, interpolation="nearest")
                status = "🟢 ALIVE" if self.result.observation.alive else "🔴 DEAD"
                ax_game.set_title(
                    f"Snake Game - {status}\n"
                    f"Step: {steps}, Score: {self.result.observation.episode_score:.1f}, "
                    f"Fruits: {self.result.observation.episode_fruits}",
                    fontsize=12,
                    fontweight="bold",
                )

                # Stats
                ax_stats.plot(scores_history, "g-", linewidth=2, label="Score")
                ax_stats.set_title("Score Over Time")
                ax_stats.set_xlabel("Step")
                ax_stats.set_ylabel("Cumulative Score")
                ax_stats.grid(True, alpha=0.3)
                ax_stats.legend()

                plt.pause(0.01)

        plt.ioff()
        plt.show()

        print(f"\n  Episode finished:")
        print(f"    Steps: {self.result.observation.episode_steps}")
        print(f"    Score: {self.result.observation.episode_score:.2f}")
        print(f"    Fruits: {self.result.observation.episode_fruits}")

    def play_multiple_episodes(self, num_episodes=5, max_steps_per_episode=200):
        """Play multiple episodes and show statistics."""
        print("\n" + "=" * 70)
        print(f"  Playing {num_episodes} Episodes")
        print("=" * 70)

        all_scores = []
        all_fruits = []
        all_steps = []

        for ep in range(num_episodes):
            self.reset_game()
            steps = 0

            while not self.result.done and steps < max_steps_per_episode:
                import random

                action = random.randint(0, 2)
                self.step_game(action)
                steps += 1

            all_scores.append(self.result.observation.episode_score)
            all_fruits.append(self.result.observation.episode_fruits)
            all_steps.append(self.result.observation.episode_steps)

            print(
                f"  Episode {ep + 1}/{num_episodes}: "
                f"Steps={steps}, Score={self.result.observation.episode_score:.2f}, "
                f"Fruits={self.result.observation.episode_fruits}"
            )

        # Show statistics
        print("\n" + "=" * 70)
        print("  Statistics Across All Episodes")
        print("=" * 70)
        print(f"  Average Score:  {np.mean(all_scores):.2f} ± {np.std(all_scores):.2f}")
        print(f"  Best Score:     {np.max(all_scores):.2f}")
        print(f"  Average Fruits: {np.mean(all_fruits):.1f} ± {np.std(all_fruits):.1f}")
        print(f"  Best Fruits:    {np.max(all_fruits)}")
        print(f"  Average Steps:  {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")

        # Visualize statistics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].bar(range(1, num_episodes + 1), all_scores, color="green", alpha=0.7)
        axes[0].set_title("Scores per Episode")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Score")
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(range(1, num_episodes + 1), all_fruits, color="red", alpha=0.7)
        axes[1].set_title("Fruits Collected per Episode")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Fruits")
        axes[1].grid(True, alpha=0.3)

        axes[2].bar(range(1, num_episodes + 1), all_steps, color="blue", alpha=0.7)
        axes[2].set_title("Steps per Episode")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Steps")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Main function."""
    print("=" * 70)
    print("  Snake Game - Interactive Player")
    print("=" * 70)

    parser = argparse.ArgumentParser(
        description="Play the snake game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Docker mode - automated play
  python examples/snake_rl/play_snake.py --mode docker --play-mode auto

  # Local server mode - multiple episodes
  python examples/snake_rl/play_snake.py --mode local --play-mode multi --episodes 10
        """,
    )

    # Connection mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["docker", "local"],
        default="docker",
        help="Connection mode: docker or local",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Server URL for local mode",
    )

    # Play mode
    parser.add_argument(
        "--play-mode",
        type=str,
        default="auto",
        choices=["auto", "multi"],
        help="Play mode: auto (single episode) or multi (multiple episodes)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "simple"],
        help="Agent policy for automated play",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes for multi mode"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Maximum steps per episode"
    )

    args = parser.parse_args()

    if args.mode == "docker":
        print("\n📦 Mode: Docker")
        print("  Running containerized environment...")
    else:
        print(f"\n🖥️  Mode: Local Server ({args.url})")

    try:
        if args.mode == "docker":
            print("\nStarting Docker container...")
            client = SnakeEnv.from_docker_image("snake-env:latest")
            print("✓ Container started successfully!\n")
        else:
            print(f"\nConnecting to {args.url}...")
            client = SnakeEnv(base_url=args.url)
            client.reset()  # Test connection
            print("✓ Connected successfully!\n")

        player = SnakeGamePlayer(client)

        if args.play_mode == "auto":
            player.play_automated(max_steps=args.steps, policy=args.policy)
        elif args.play_mode == "multi":
            player.play_multiple_episodes(
                num_episodes=args.episodes, max_steps_per_episode=args.steps
            )

        print("\n" + "=" * 70)
        print("  Game Session Complete!")
        print("=" * 70)

        # Cleanup
        print("\nCleaning up...")
        if args.mode == "docker":
            client.close()
            print("✓ Container stopped and removed")
        else:
            print("✓ Disconnected from server")

        return True

    except Exception as e:
        print(f"\n❌ Game failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
