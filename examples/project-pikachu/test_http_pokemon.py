#!/usr/bin/env python3
"""
HTTP Client Test Script for Pokemon Environment.

This script tests the Pokemon environment via HTTP client (the OpenEnv way).

Prerequisites:
1. Pokemon environment server running (either locally or in Docker)
   - Local: python -m poke_env.server.app
   - Docker: docker run -p 8000:8000 -p 9980:9980 pokemon-env:latest

2. Server accessible at http://localhost:9980

Usage:
    # Test against local server
    python test_http_pokemon.py

    # Test against custom URL
    python test_http_pokemon.py --url http://localhost:9980
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from envs.pokemon_env.client import PokemonEnv
from envs.pokemon_env.models import PokemonAction


def test_health_check(base_url: str):
    """Test 1: Is the server healthy?"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)

    try:
        import requests
        response = requests.get(f"{base_url}/health", timeout=5)

        if response.status_code == 200:
            print(f"✅ Server is healthy!")
            print(f"   Status: {response.json()}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_client_creation(base_url: str):
    """Test 2: Can we create a client?"""
    print("\n" + "="*80)
    print("TEST 2: Client Creation")
    print("="*80)

    try:
        client = PokemonEnv(base_url=base_url)
        print("✅ Client created successfully")
        print(f"   Base URL: {client.base_url}")
        return True
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return False


def test_reset_via_http(base_url: str):
    """Test 3: Can we reset via HTTP?"""
    print("\n" + "="*80)
    print("TEST 3: HTTP Reset")
    print("="*80)

    try:
        client = PokemonEnv(base_url=base_url)
        print("Calling client.reset()...")

        result = client.reset()

        print("✅ Reset successful!")
        print(f"   Reward: {result.reward}")
        print(f"   Done: {result.done}")

        obs = result.observation
        print(f"   Turn: {obs.turn}")

        if obs.active_pokemon:
            print(f"   Active Pokemon: {obs.active_pokemon.species} (HP: {obs.active_pokemon.hp_percent*100:.1f}%)")
        if obs.opponent_active_pokemon:
            print(f"   Opponent: {obs.opponent_active_pokemon.species} (HP: {obs.opponent_active_pokemon.hp_percent*100:.1f}%)")

        print(f"   Available moves: {len(obs.available_moves)}")
        print(f"   Available switches: {len(obs.available_switches)}")

        return True
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_via_http(base_url: str):
    """Test 4: Can we step via HTTP?"""
    print("\n" + "="*80)
    print("TEST 4: HTTP Step")
    print("="*80)

    try:
        client = PokemonEnv(base_url=base_url)
        print("Resetting...")
        result = client.reset()

        print("Taking action: move index 0")
        action = PokemonAction(action_type="move", action_index=0)

        print("Calling client.step()...")
        result = client.step(action)

        print("✅ Step successful!")
        obs = result.observation
        print(f"   Turn: {obs.turn}")
        print(f"   Reward: {result.reward}")
        print(f"   Done: {result.done}")

        if obs.active_pokemon:
            print(f"   Active Pokemon: {obs.active_pokemon.species} (HP: {obs.active_pokemon.hp_percent*100:.1f}%)")

        return True
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_battle_via_http(base_url: str):
    """Test 5: Can we complete a battle via HTTP?"""
    print("\n" + "="*80)
    print("TEST 5: Full Battle via HTTP")
    print("="*80)

    try:
        client = PokemonEnv(base_url=base_url)
        print("Resetting...")
        result = client.reset()

        print("Starting battle loop...")
        max_turns = 100
        turn = 0

        while not result.done and turn < max_turns:
            turn += 1
            obs = result.observation

            # Choose action
            if obs.available_moves:
                action = PokemonAction(action_type="move", action_index=0)
            elif obs.available_switches:
                action = PokemonAction(action_type="switch", action_index=0)
            else:
                print("   No legal actions!")
                break

            print(f"   Turn {turn}: {action.action_type} {action.action_index}", end="")

            result = client.step(action)
            obs = result.observation

            if obs.active_pokemon and obs.opponent_active_pokemon:
                print(f" | Us: {obs.active_pokemon.species} ({obs.active_pokemon.hp_percent*100:.0f}%)", end="")
                print(f" vs Opp: {obs.opponent_active_pokemon.species} ({obs.opponent_active_pokemon.hp_percent*100:.0f}%)")
            else:
                print()

        print(f"\n✅ Battle completed after {turn} turns!")
        print(f"   Final reward: {result.reward}")
        print(f"   Done: {result.done}")

        # Check state
        state = client.state()
        print(f"   Battle ID: {state.battle_id}")
        print(f"   Winner: {state.battle_winner}")

        return True
    except Exception as e:
        print(f"❌ Battle failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_endpoint(base_url: str):
    """Test 6: Can we query state?"""
    print("\n" + "="*80)
    print("TEST 6: State Endpoint")
    print("="*80)

    try:
        client = PokemonEnv(base_url=base_url)
        result = client.reset()

        # Take a few steps
        for _ in range(3):
            if result.done:
                break
            action = PokemonAction(action_type="move", action_index=0)
            result = client.step(action)

        # Query state
        state = client.state()

        print("✅ State endpoint working!")
        print(f"   Episode ID: {state.episode_id}")
        print(f"   Step count: {state.step_count}")
        print(f"   Battle ID: {state.battle_id}")
        print(f"   Format: {state.battle_format}")
        print(f"   Player: {state.player_username}")
        print(f"   Finished: {state.is_battle_finished}")

        return True
    except Exception as e:
        print(f"❌ State query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all HTTP tests."""
    parser = argparse.ArgumentParser(description="Test Pokemon environment via HTTP")
    parser.add_argument("--url", default="http://localhost:9980", help="Server URL")
    args = parser.parse_args()

    print("="*80)
    print("Pokemon Environment HTTP Test Suite")
    print("="*80)
    print(f"\nTesting server at: {args.url}")
    print("Make sure the server is running!\n")

    input("Press Enter to start tests...")

    results = []

    # Run all tests
    results.append(("Health Check", test_health_check(args.url)))
    results.append(("Client Creation", test_client_creation(args.url)))
    results.append(("HTTP Reset", test_reset_via_http(args.url)))
    results.append(("HTTP Step", test_step_via_http(args.url)))
    results.append(("Full Battle", test_full_battle_via_http(args.url)))
    results.append(("State Endpoint", test_state_endpoint(args.url)))

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
        print("\n🎉 All tests passed! HTTP client is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
