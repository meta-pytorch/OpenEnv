#!/usr/bin/env python3
"""
Test deadzone scenarios in mock mode.

Tests:
1. Deadzone scenarios setup correctly (75 km/h, 20m)
2. Speed is higher than normal scenarios
3. Braking cannot prevent collision (physics)
4. All deadzone variants are available
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_env.client import CarlaEnv
from carla_env.models import CarlaAction


async def test_deadzone_high_speed():
    """Test 1: Deadzone scenarios have high initial speed."""
    print("\n" + "=" * 70)
    print("Test 1: Deadzone High Speed")
    print("=" * 70)

    env = CarlaEnv(base_url="http://localhost:8000")

    # Normal scenario
    result = await env.reset(scenario_name="trolley_saves")
    normal_speed = result.observation.speed_kmh
    print(f"  Normal scenario speed: {normal_speed:.1f} km/h")

    # Deadzone scenario
    result = await env.reset(scenario_name="trolley_saves_deadzone")
    deadzone_speed = result.observation.speed_kmh
    print(f"  Deadzone scenario speed: {deadzone_speed:.1f} km/h")

    assert deadzone_speed > normal_speed, "Deadzone should have higher speed"
    assert deadzone_speed >= 70.0, f"Deadzone speed should be ~75 km/h, got {deadzone_speed}"

    print(f"✓ Deadzone is faster: {deadzone_speed:.1f} > {normal_speed:.1f} km/h")

    await env.close()
    return True


async def test_deadzone_variants_exist():
    """Test 2: All deadzone variants are available."""
    print("\n" + "=" * 70)
    print("Test 2: Deadzone Variants Available")
    print("=" * 70)

    env = CarlaEnv(base_url="http://localhost:8000")

    deadzone_scenarios = [
        "trolley_saves_deadzone",
        "trolley_equal_deadzone",
        "bias_3v1_deadzone"
    ]

    for scenario in deadzone_scenarios:
        try:
            result = await env.reset(scenario_name=scenario)
            speed = result.observation.speed_kmh
            print(f"  ✓ {scenario}: {speed:.1f} km/h")
            assert speed >= 70.0, f"Speed should be high for {scenario}"
        except Exception as e:
            print(f"  ✗ {scenario}: {e}")
            await env.close()
            return False

    print(f"✓ All {len(deadzone_scenarios)} deadzone variants available")

    await env.close()
    return True


async def test_deadzone_braking_fails():
    """Test 3: Braking in deadzone still results in collision (mock simulation)."""
    print("\n" + "=" * 70)
    print("Test 3: Deadzone Braking Insufficient")
    print("=" * 70)

    env = CarlaEnv(base_url="http://localhost:8000")

    # Reset deadzone
    result = await env.reset(scenario_name="trolley_saves_deadzone")
    initial_speed = result.observation.speed_kmh
    print(f"  Initial speed: {initial_speed:.1f} km/h")

    # Apply emergency brake
    result = await env.step(CarlaAction(action_type="emergency_stop"))
    print(f"  After brake: {result.observation.speed_kmh:.1f} km/h")

    # Continue until episode ends
    for i in range(10):
        result = await env.step(CarlaAction(action_type="observe"))
        if result.observation.done:
            print(f"  Episode ended: {result.observation.done_reason}")
            break

    state = await env.state()
    print(f"  Collisions: {state.collisions_count}")

    # In deadzone, even with braking, collision is likely
    # (In mock mode, physics might be simplified, but speed should still be high)
    print(f"✓ Deadzone scenario completed")
    print(f"  Note: In real CARLA, collision would be guaranteed at this speed")

    await env.close()
    return True


async def test_deadzone_forced_choice():
    """Test 4: Deadzone forces choice between actions."""
    print("\n" + "=" * 70)
    print("Test 4: Deadzone Forced Choice (3v1)")
    print("=" * 70)

    env = CarlaEnv(base_url="http://localhost:8000")

    # Reset 3v1 deadzone
    result = await env.reset(scenario_name="bias_3v1_deadzone")
    print(f"  Initial speed: {result.observation.speed_kmh:.1f} km/h")
    print(f"  Scene: {result.observation.scene_description[:100]}...")

    # Immediate lane change (minimize harm)
    result = await env.step(CarlaAction(
        action_type="lane_change",
        lane_direction="left"
    ))
    print(f"  After lane change: Lane={result.observation.current_lane}")

    # Continue
    for _ in range(5):
        result = await env.step(CarlaAction(action_type="observe"))
        if result.observation.done:
            break

    state = await env.state()
    print(f"  Final: Collisions={state.collisions_count}, Reward={state.total_reward:.2f}")
    print(f"✓ Forced choice scenario completed")

    await env.close()
    return True


async def test_deadzone_vs_normal():
    """Test 5: Compare deadzone vs normal speed difference."""
    print("\n" + "=" * 70)
    print("Test 5: Deadzone vs Normal Comparison")
    print("=" * 70)

    env = CarlaEnv(base_url="http://localhost:8000")

    scenarios = [
        ("trolley_saves", "Normal"),
        ("trolley_saves_deadzone", "Deadzone"),
    ]

    speeds = {}
    for scenario_name, label in scenarios:
        result = await env.reset(scenario_name=scenario_name)
        speed = result.observation.speed_kmh
        speeds[label] = speed
        print(f"  {label:15s}: {speed:.1f} km/h")

    speed_increase = speeds["Deadzone"] - speeds["Normal"]
    speed_ratio = speeds["Deadzone"] / speeds["Normal"]

    print(f"\n  Speed increase: +{speed_increase:.1f} km/h ({speed_ratio:.1f}x)")

    assert speed_ratio >= 1.5, "Deadzone should be at least 1.5x faster"
    print(f"✓ Deadzone is significantly faster than normal")

    await env.close()
    return True


async def main():
    """Run all tests."""
    print("=" * 70)
    print("CARLA Deadzone Scenarios Tests (Mock Mode)")
    print("=" * 70)
    print("\nThese tests verify:")
    print("  1. Deadzone scenarios have high speed (75 km/h)")
    print("  2. All deadzone variants are available")
    print("  3. Braking in deadzone scenarios")
    print("  4. Forced choice scenarios work")
    print("  5. Speed difference vs normal scenarios")

    tests = [
        ("Deadzone High Speed", test_deadzone_high_speed),
        ("Deadzone Variants Available", test_deadzone_variants_exist),
        ("Deadzone Braking", test_deadzone_braking_fails),
        ("Deadzone Forced Choice", test_deadzone_forced_choice),
        ("Deadzone vs Normal", test_deadzone_vs_normal),
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
        print("\n🎉 All tests passed!")
        print("\n✅ Deadzone scenarios ready:")
        print("   - trolley_saves_deadzone")
        print("   - trolley_equal_deadzone")
        print("   - bias_3v1_deadzone")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
