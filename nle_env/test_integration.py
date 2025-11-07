#!/usr/bin/env python3
"""
Basic integration test for NLE environment.

This script tests that the NLE environment can be imported and basic
structure is correct. Does NOT require NLE to be installed (that happens
in Docker).
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

print("=" * 70)
print("NLE Environment Integration Test")
print("=" * 70)

# Test 1: Import models
print("\n[1/5] Testing model imports...")
try:
    from envs.nle_env import NLEAction, NLEObservation, NLEState

    print("✓ Models imported successfully")
    print(f"  - NLEAction: {NLEAction}")
    print(f"  - NLEObservation: {NLEObservation}")
    print(f"  - NLEState: {NLEState}")
except Exception as e:
    print(f"✗ Failed to import models: {e}")
    sys.exit(1)

# Test 2: Import client
print("\n[2/5] Testing client import...")
try:
    from envs.nle_env import NLEEnv

    print("✓ Client imported successfully")
    print(f"  - NLEEnv: {NLEEnv}")
except Exception as e:
    print(f"✗ Failed to import client: {e}")
    sys.exit(1)

# Test 3: Create action instances
print("\n[3/5] Testing action creation...")
try:
    action1 = NLEAction(action_id=0)  # Move north
    action2 = NLEAction(action_id=37)  # Eat
    action3 = NLEAction(action_id=50)  # Search

    print("✓ Actions created successfully")
    print(f"  - Move north: {action1}")
    print(f"  - Eat: {action2}")
    print(f"  - Search: {action3}")
except Exception as e:
    print(f"✗ Failed to create actions: {e}")
    sys.exit(1)

# Test 4: Create observation instances
print("\n[4/5] Testing observation creation...")
try:
    obs = NLEObservation(
        glyphs=[[0] * 79 for _ in range(21)],
        blstats=[0] * 26,
        message=[0] * 256,
        done=False,
        reward=0.0,
    )

    print("✓ Observation created successfully")
    print(f"  - done: {obs.done}")
    print(f"  - reward: {obs.reward}")
    print(f"  - glyphs shape: {len(obs.glyphs)}x{len(obs.glyphs[0])}")
    print(f"  - blstats length: {len(obs.blstats)}")
except Exception as e:
    print(f"✗ Failed to create observation: {e}")
    sys.exit(1)

# Test 5: Create state instances
print("\n[5/5] Testing state creation...")
try:
    state = NLEState(
        episode_id="test_123",
        step_count=42,
        game_over=False,
        end_status="RUNNING",
        in_normal_game=True,
        character="mon-hum-neu-mal",
        task_name="NetHackScore-v0",
    )

    print("✓ State created successfully")
    print(f"  - episode_id: {state.episode_id}")
    print(f"  - step_count: {state.step_count}")
    print(f"  - end_status: {state.end_status}")
    print(f"  - character: {state.character}")
except Exception as e:
    print(f"✗ Failed to create state: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All basic integration tests passed!")
print("=" * 70)
print("\nNext steps:")
print("  1. Build Docker image (from repo root):")
print("     cd /Users/sanyambhutani/GH/OpenEnv")
print("     docker build -f src/envs/nle_env/server/Dockerfile -t nle-env:latest .")
print("  2. Run server: docker run -p 8000:8000 nle-env:latest")
print("  3. Test client: python examples/test_nle_client.py")
print()
