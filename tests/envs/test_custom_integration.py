"""Integration test for custom BrowserGym environment.

This test verifies the end-to-end functionality:
1. Environment can be created with custom tasks
2. HTML content loads correctly
3. Actions can be executed
4. Observations are returned properly
5. Rewards are calculated

Run this test:
    source .venv/bin/activate
    python3 tests/envs/test_custom_integration.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def test_custom_env_creation():
    """Test that custom environment can be created."""
    print("Testing custom environment creation...")
    try:
        from envs.browsergym_env.server.browsergym_environment import (
            BrowserGymEnvironment,
        )

        env = BrowserGymEnvironment(
            benchmark="custom",
            task_name="copy-paste",
            headless=True,
            viewport_width=1280,
            viewport_height=720,
        )

        print("  PASS: Custom environment created successfully")
        return env

    except Exception as e:
        print(f"  FAIL: Failed to create environment: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_env_reset(env):
    """Test that environment can be reset."""
    print("\nTesting environment reset...")
    try:
        obs = env.reset()

        # Verify observation structure
        assert hasattr(obs, "goal"), "Observation missing 'goal' field"
        assert hasattr(obs, "text"), "Observation missing 'text' field"
        assert hasattr(obs, "done"), "Observation missing 'done' field"
        assert hasattr(obs, "reward"), "Observation missing 'reward' field"

        print("  PASS: Environment reset successful")
        print(f"  Goal: {obs.goal[:60]}...")
        print(f"  Page text length: {len(obs.text)} characters")
        print(f"  Initial done: {obs.done}")

        return True

    except Exception as e:
        print(f"  FAIL: Reset failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_env_step(env):
    """Test that actions can be executed."""
    print("\nTesting environment step...")
    try:
        from envs.browsergym_env.models import BrowserGymAction

        # Try a simple action (click on source text)
        action = BrowserGymAction(action_str="click('#source-text')")
        obs = env.step(action)

        # Verify observation
        assert hasattr(obs, "reward"), "Observation missing 'reward' field"
        assert hasattr(obs, "done"), "Observation missing 'done' field"

        print("  PASS: Step executed successfully")
        print(f"  Reward: {obs.reward}")
        print(f"  Done: {obs.done}")

        return True

    except Exception as e:
        print(f"  FAIL: Step failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_env_state(env):
    """Test that environment state is accessible."""
    print("\nTesting environment state...")
    try:
        state = env.state

        # Verify state structure
        assert hasattr(state, "benchmark"), "State missing 'benchmark' field"
        assert hasattr(state, "task_name"), "State missing 'task_name' field"
        assert hasattr(state, "step_count"), "State missing 'step_count' field"

        assert state.benchmark == "custom", f"Expected benchmark='custom', got '{state.benchmark}'"
        assert (
            state.task_name == "copy-paste"
        ), f"Expected task_name='copy-paste', got '{state.task_name}'"

        print("  PASS: State accessible")
        print(f"  Benchmark: {state.benchmark}")
        print(f"  Task: {state.task_name}")
        print(f"  Step count: {state.step_count}")

        return True

    except Exception as e:
        print(f"  FAIL: State access failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_task_html_loading():
    """Test that task HTML is loaded correctly."""
    print("\nTesting HTML task loading...")
    try:
        from envs.browsergym_env.server.custom.custom_tasks import get_custom_task

        task = get_custom_task("copy-paste")

        # Verify task has URL
        url = task._get_task_url()
        assert url, "Task URL is empty"
        print(f"  PASS: Task URL: {url[:80]}...")

        # Verify goal description
        goal = task._get_goal_description()
        assert goal, "Goal description is empty"
        assert len(goal) > 0, "Goal description is too short"
        print(f"  PASS: Goal: {goal[:80]}...")

        return True

    except Exception as e:
        print(f"  FAIL: HTML loading test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multitab_task():
    """Test that multi-tab task is available."""
    print("\nTesting multi-tab task...")
    try:
        from envs.browsergym_env.server.custom.custom_tasks import get_custom_task

        task = get_custom_task("copy-paste-multitab")
        url = task._get_task_url()
        goal = task._get_goal_description()

        assert url, "Multi-tab task URL is empty"
        assert goal, "Multi-tab task goal is empty"

        print("  PASS: Multi-tab task available")
        print(f"  Goal: {goal[:80]}...")

        return True

    except Exception as e:
        print(f"  FAIL: Multi-tab task test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Custom BrowserGym Environment - Integration Tests")
    print("=" * 70)
    print("\nThese tests verify the custom environment works end-to-end.")
    print("This may take a minute as it starts a headless browser...\n")

    # Test 1: HTML task loading (no browser needed)
    test_results = []
    test_results.append(("HTML Task Loading", test_task_html_loading()))
    test_results.append(("Multi-tab Task", test_multitab_task()))

    # Test 2-5: Full environment tests (requires browser)
    env = test_custom_env_creation()
    if env:
        test_results.insert(0, ("Environment Creation", True))
        test_results.append(("Environment Reset", test_env_reset(env)))
        test_results.append(("Environment Step", test_env_step(env)))
        test_results.append(("Environment State", test_env_state(env)))

        # Cleanup
        try:
            env.close()
            print("\nEnvironment closed successfully")
        except Exception as e:
            print(f"\nEnvironment cleanup warning: {e}")
    else:
        test_results.insert(0, ("Environment Creation", False))
        print("\nSkipping browser tests due to environment creation failure")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED")
        print("\nCustom BrowserGym environment is fully functional:")
        print("  - HTML tasks load correctly")
        print("  - Environment follows OpenEnv interface")
        print("  - Actions execute properly")
        print("  - Observations are returned correctly")
        print("  - State is accessible")
    else:
        print("Some integration tests failed.")
        print("Please review the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
