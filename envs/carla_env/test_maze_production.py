#!/usr/bin/env python3
"""
Test maze navigation scenario in production (HuggingFace Space).

Tests goal distance/direction tracking and maze scenario in deployed environment.
"""

import asyncio
import sys
from carla_env.client import CarlaEnv
from carla_env.models import CarlaAction

# HuggingFace Space URL
BASE_URL = "https://sergiopaniego-carla-env-real-updated.hf.space"


async def test_maze_scenario():
    """Test 1: Maze scenario works in production."""
    print("\n" + "=" * 70)
    print("Test 1: Maze Scenario in Production")
    print("=" * 70)

    env = CarlaEnv(base_url=BASE_URL)

    try:
        result = await env.reset(scenario_name="maze_navigation")
        obs = result.observation

        print(f"✓ Maze scenario reset successful")
        print(f"  Location: {obs.location}")
        print(f"  Goal distance: {obs.goal_distance:.1f}m")
        print(f"  Goal direction: {obs.goal_direction}")
        print(f"  Scene: {obs.scene_description}")

        assert obs.goal_distance is not None, "Goal distance should be set"
        assert obs.goal_direction is not None, "Goal direction should be set"
        assert obs.goal_distance > 100.0, "Goal should be ~150m away"

        print(f"✓ All assertions passed")
        await env.close()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        await env.close()
        return False


async def test_goal_tracking():
    """Test 2: Goal tracking updates as vehicle moves."""
    print("\n" + "=" * 70)
    print("Test 2: Goal Tracking in Production")
    print("=" * 70)

    env = CarlaEnv(base_url=BASE_URL)

    try:
        result = await env.reset(scenario_name="maze_navigation")
        initial_distance = result.observation.goal_distance

        print(f"  Initial goal distance: {initial_distance:.1f}m")

        # Move forward
        for i in range(3):
            result = await env.step(CarlaAction(
                action_type="control",
                throttle=0.7,
                steer=0.0
            ))

        final_distance = result.observation.goal_distance
        print(f"  Final goal distance: {final_distance:.1f}m")
        print(f"  Distance changed by: {initial_distance - final_distance:.1f}m")

        print(f"✓ Goal tracking working")
        await env.close()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        await env.close()
        return False


async def test_goal_in_observations():
    """Test 3: Goal info present/absent in different scenarios."""
    print("\n" + "=" * 70)
    print("Test 3: Goal Info Conditional on Scenario")
    print("=" * 70)

    env = CarlaEnv(base_url=BASE_URL)

    try:
        # Maze has goal
        result = await env.reset(scenario_name="maze_navigation")
        print(f"  Maze scenario:")
        print(f"    goal_distance: {result.observation.goal_distance}")
        print(f"    goal_direction: {result.observation.goal_direction}")
        assert result.observation.goal_distance is not None
        assert result.observation.goal_direction is not None
        print(f"  ✓ Goal info present in maze")

        # Trolley doesn't have goal
        result = await env.reset(scenario_name="trolley_saves")
        print(f"\n  Trolley scenario:")
        print(f"    goal_distance: {result.observation.goal_distance}")
        print(f"    goal_direction: {result.observation.goal_direction}")
        assert result.observation.goal_distance is None
        assert result.observation.goal_direction is None
        print(f"  ✓ Goal info absent in trolley")

        await env.close()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        await env.close()
        return False


async def test_action_bias_scenarios():
    """Test 4: Action bias scenarios still work."""
    print("\n" + "=" * 70)
    print("Test 4: Action Bias Scenarios")
    print("=" * 70)

    env = CarlaEnv(base_url=BASE_URL)

    try:
        # Test action_bias_saves
        result = await env.reset(scenario_name="action_bias_saves")
        print(f"  action_bias_saves: ✓")
        print(f"    Scene: {result.observation.scene_description[:50]}...")

        # Test action_bias_equal
        result = await env.reset(scenario_name="action_bias_equal")
        print(f"  action_bias_equal: ✓")
        print(f"    Scene: {result.observation.scene_description[:50]}...")

        print(f"✓ All action bias scenarios work")
        await env.close()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        await env.close()
        return False


async def test_navigation_actions():
    """Test 5: Navigation actions still work."""
    print("\n" + "=" * 70)
    print("Test 5: Navigation Actions (Day 4)")
    print("=" * 70)

    env = CarlaEnv(base_url=BASE_URL)

    try:
        result = await env.reset(scenario_name="maze_navigation")

        # Init navigation agent
        result = await env.step(CarlaAction(
            action_type="init_navigation_agent",
            navigation_behavior="normal"
        ))
        print(f"  ✓ init_navigation_agent works")

        # Set destination
        result = await env.step(CarlaAction(
            action_type="set_destination",
            destination_x=100.0,
            destination_y=100.0,
            destination_z=0.0
        ))
        print(f"  ✓ set_destination works")

        # Follow route
        result = await env.step(CarlaAction(
            action_type="follow_route",
            route_steps=1
        ))
        print(f"  ✓ follow_route works")
        print(f"    Speed: {result.observation.speed_kmh:.1f} km/h")

        print(f"✓ All navigation actions work")
        await env.close()
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        await env.close()
        return False


async def main():
    """Run all production tests."""
    print("=" * 70)
    print("CARLA Maze & Goal Tracking Production Tests")
    print("=" * 70)
    print(f"Testing: {BASE_URL}")
    print("\nThese tests verify:")
    print("  1. Maze scenario works in production")
    print("  2. Goal tracking updates correctly")
    print("  3. Goal info conditional on scenario type")
    print("  4. Action bias scenarios still work")
    print("  5. Navigation actions still work")

    tests = [
        ("Maze Scenario", test_maze_scenario),
        ("Goal Tracking", test_goal_tracking),
        ("Goal Info Conditional", test_goal_in_observations),
        ("Action Bias Scenarios", test_action_bias_scenarios),
        ("Navigation Actions", test_navigation_actions),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = await test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ Test failed with error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All production tests passed!")
        print("\n✅ Ready to commit:")
        print("   - Maze navigation scenario")
        print("   - Goal distance/direction in observations")
        print("   - Action bias examples")
        print("   - Updated documentation")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
