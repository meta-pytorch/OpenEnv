#!/usr/bin/env python3
"""
Simple example showing how to use the FinRL environment with OpenEnv.

This example demonstrates:
1. Connecting to a FinRL environment server
2. Resetting the environment
3. Executing random trading actions
4. Tracking portfolio value over time
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.finrl_env import FinRLAction, FinRLEnv


def main():
    """Run a simple FinRL environment example."""
    print("=" * 70)
    print("FinRL Environment - Simple Example")
    print("=" * 70)
    print()

    # Connect to server
    print("Connecting to FinRL environment at http://localhost:8000...")
    try:
        client = FinRLEnv(base_url="http://localhost:8000")
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        print()
        print("Make sure the server is running:")
        print("  docker run -p 8000:8000 finrl-env:latest")
        return False

    print("✅ Connected successfully!")
    print()

    # Get configuration
    try:
        config = client.get_config()
        print("Environment Configuration:")
        print(f"  Stock dimension: {config['stock_dim']}")
        print(f"  Initial amount: ${config['initial_amount']:,.0f}")
        print(f"  Action space: {config['action_space']}")
        print(f"  State space: {config['state_space']}")
        print(f"  Technical indicators: {', '.join(config['tech_indicators'])}")
        print()
    except Exception as e:
        print(f"⚠️  Could not fetch config: {e}")
        print()

    # Reset environment
    print("Resetting environment...")
    result = client.reset()
    print(f"✅ Environment reset successfully!")
    print(f"   Initial portfolio value: ${result.observation.portfolio_value:,.2f}")
    print(f"   State dimension: {len(result.observation.state)}")
    if result.observation.date:
        print(f"   Starting date: {result.observation.date}")
    print()

    # Run trading simulation
    print("-" * 70)
    print("Running 20-step trading simulation with random actions...")
    print("-" * 70)
    print()

    portfolio_history = [result.observation.portfolio_value]
    cumulative_reward = 0

    for step in range(20):
        # Get current state
        state = result.observation.state

        # Generate random actions (in real use, replace with your RL policy)
        num_stocks = config.get("stock_dim", 1)
        actions = np.random.uniform(-0.5, 0.5, size=num_stocks).tolist()

        # Execute action
        result = client.step(FinRLAction(actions=actions))

        # Track metrics
        portfolio_history.append(result.observation.portfolio_value)
        cumulative_reward += result.reward or 0

        # Print progress
        print(
            f"Step {step + 1:2d}: "
            f"Portfolio=${result.observation.portfolio_value:>12,.2f} | "
            f"Reward={result.reward:>8.2f} | "
            f"Date={result.observation.date}"
        )

        if result.done:
            print()
            print("Episode finished!")
            break

    # Summary
    print()
    print("-" * 70)
    print("Trading Simulation Complete")
    print("-" * 70)
    print(f"Initial portfolio value: ${portfolio_history[0]:,.2f}")
    print(f"Final portfolio value:   ${portfolio_history[-1]:,.2f}")
    print(
        f"Total return:            ${portfolio_history[-1] - portfolio_history[0]:,.2f} "
        f"({((portfolio_history[-1] / portfolio_history[0] - 1) * 100):.2f}%)"
    )
    print(f"Cumulative reward:       {cumulative_reward:.2f}")
    print(f"Steps executed:          {len(portfolio_history) - 1}")
    print()

    # Plot portfolio value over time (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_history, marker="o", linewidth=2)
        plt.title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("finrl_portfolio_history.png", dpi=150)
        print("📊 Portfolio chart saved to: finrl_portfolio_history.png")
        print()
    except ImportError:
        pass

    # Cleanup
    print("Closing connection...")
    client.close()
    print("✅ Done!")
    print()

    print("=" * 70)
    print("Example completed successfully! 🎉")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
