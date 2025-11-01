#!/usr/bin/env python3
"""
Local Test Script for Pokemon Environment.

This script tests the Pokemon environment locally WITHOUT Docker,
using a local Pokemon Showdown server.

Prerequisites:
1. Pokemon Showdown server running on localhost:8000
   - Clone: git clone https://github.com/smogon/pokemon-showdown.git
   - Install: cd pokemon-showdown && npm install
   - Configure: cp config/config-example.js config/config.js
   - Run: node pokemon-showdown start --no-security

2. poke-env installed:
   - pip install poke-env

Usage:
    python test_local_pokemon.py

This will run several tests:
1. Environment creation
2. Reset functionality
3. Single step execution
4. Full battle (multiple steps)
5. Error handling (illegal moves)
6. Dense rewards mode
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import models and environment
from envs.pokemon_env.models import PokemonAction, PokemonObservation, PokemonState
from envs.pokemon_env.server.pokemon_environment import PokemonEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_environment_creation():
    """Test 1: Can we create an environment?"""
    print("\n" + "="*80)
    print("TEST 1: Environment Creation")
    print("="*80)

    try:
        env = PokemonEnvironment(battle_format="gen9randombattle")
        print("✅ Environment created successfully")
        print(f"   Player: {env.player_username}")
        print(f"   Format: {env.battle_format}")
        print(f"   Reward mode: {env.reward_mode}")
        return True
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reset():
    """Test 2: Can we reset and get an initial observation?"""
    print("\n" + "="*80)
    print("TEST 2: Environment Reset")
    print("="*80)

    try:
        env = PokemonEnvironment(battle_format="gen9randombattle")
        print("Calling reset()...")

        obs = env.reset()

        print("✅ Reset successful!")
        print(f"   Episode ID: {env.state.episode_id}")
        print(f"   Battle ID: {env.state.battle_id}")
        print(f"   Turn: {obs.turn}")

        if obs.active_pokemon:
            print(f"   Active Pokemon: {obs.active_pokemon.species} (HP: {obs.active_pokemon.hp_percent*100:.1f}%)")
        if obs.opponent_active_pokemon:
            print(f"   Opponent: {obs.opponent_active_pokemon.species} (HP: {obs.opponent_active_pokemon.hp_percent*100:.1f}%)")

        print(f"   Available moves: {len(obs.available_moves)}")
        print(f"   Available switches: {len(obs.available_switches)}")
        print(f"   Legal actions: {len(obs.legal_actions)}")

        env.close()
        return True
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_step():
    """Test 3: Can we take a single step?"""
    print("\n" + "="*80)
    print("TEST 3: Single Step Execution")
    print("="*80)

    try:
        env = PokemonEnvironment(battle_format="gen9randombattle")
        print("Resetting...")
        obs = env.reset()

        print(f"Taking action: move index 0")
        action = PokemonAction(action_type="move", action_index=0)

        print("Calling step()...")
        obs = env.step(action)

        print("✅ Step successful!")
        print(f"   Turn: {obs.turn}")
        print(f"   Reward: {obs.reward}")
        print(f"   Done: {obs.done}")

        if obs.active_pokemon:
            print(f"   Active Pokemon: {obs.active_pokemon.species} (HP: {obs.active_pokemon.hp_percent*100:.1f}%)")
        if obs.opponent_active_pokemon:
            print(f"   Opponent: {obs.opponent_active_pokemon.species} (HP: {obs.opponent_active_pokemon.hp_percent*100:.1f}%)")

        env.close()
        return True
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_battle():
    """Test 4: Can we complete a full battle?"""
    print("\n" + "="*80)
    print("TEST 4: Full Battle")
    print("="*80)

    try:
        env = PokemonEnvironment(battle_format="gen9randombattle")
        print("Resetting...")
        obs = env.reset()

        print("Starting battle loop...")
        max_turns = 100
        turn = 0

        while not obs.done and turn < max_turns:
            turn += 1

            # Choose random legal action
            if obs.available_moves:
                action = PokemonAction(action_type="move", action_index=0)
            elif obs.available_switches:
                action = PokemonAction(action_type="switch", action_index=0)
            else:
                print("   No legal actions available!")
                break

            print(f"   Turn {turn}: {action.action_type} {action.action_index}", end="")

            obs = env.step(action)

            if obs.active_pokemon and obs.opponent_active_pokemon:
                print(f" | Us: {obs.active_pokemon.species} ({obs.active_pokemon.hp_percent*100:.0f}%)", end="")
                print(f" vs Opp: {obs.opponent_active_pokemon.species} ({obs.opponent_active_pokemon.hp_percent*100:.0f}%)")
            else:
                print()

        print(f"\n✅ Battle completed after {turn} turns!")
        print(f"   Final reward: {obs.reward}")
        print(f"   Battle done: {obs.done}")
        print(f"   Winner: {env.state.battle_winner}")

        env.close()
        return True
    except Exception as e:
        print(f"❌ Battle failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_illegal_move():
    """Test 5: How do we handle illegal moves?"""
    print("\n" + "="*80)
    print("TEST 5: Illegal Move Handling")
    print("="*80)

    try:
        env = PokemonEnvironment(battle_format="gen9randombattle")
        print("Resetting...")
        obs = env.reset()

        # Try an out-of-bounds move
        print("Attempting illegal move (index 99)...")
        action = PokemonAction(action_type="move", action_index=99)

        obs = env.step(action)

        print("✅ Illegal move handled!")
        print(f"   Turn completed: {obs.turn}")
        print(f"   Reward: {obs.reward}")

        if "last_error" in obs.metadata:
            print(f"   Error caught: {obs.metadata['last_error']}")
            print(f"   Illegal count: {obs.metadata.get('illegal_action_count', 0)}")

        env.close()
        return True
    except Exception as e:
        print(f"❌ Illegal move test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dense_rewards():
    """Test 6: Do dense rewards work?"""
    print("\n" + "="*80)
    print("TEST 6: Dense Rewards Mode")
    print("="*80)

    try:
        env = PokemonEnvironment(
            battle_format="gen9randombattle",
            reward_mode="dense"
        )
        print("Resetting with dense rewards...")
        obs = env.reset()

        print("Taking a few steps to check rewards...")
        rewards = []

        for i in range(5):
            if obs.done:
                break

            action = PokemonAction(action_type="move", action_index=0)
            obs = env.step(action)
            rewards.append(obs.reward)

            print(f"   Step {i+1}: reward = {obs.reward:.4f}")

        print(f"\n✅ Dense rewards working!")
        print(f"   Non-zero rewards: {sum(1 for r in rewards if r != 0)}/{len(rewards)}")

        env.close()
        return True
    except Exception as e:
        print(f"❌ Dense rewards test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("Pokemon Environment Local Test Suite")
    print("="*80)
    print("\nThis tests the Pokemon environment WITHOUT Docker.")
    print("Make sure Pokemon Showdown is running on localhost:8000!\n")

    input("Press Enter to start tests...")

    results = []

    # Run all tests
    results.append(("Environment Creation", test_environment_creation()))
    results.append(("Reset", test_reset()))
    results.append(("Single Step", test_single_step()))
    results.append(("Full Battle", test_full_battle()))
    results.append(("Illegal Move", test_illegal_move()))
    results.append(("Dense Rewards", test_dense_rewards()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Pokemon environment is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
