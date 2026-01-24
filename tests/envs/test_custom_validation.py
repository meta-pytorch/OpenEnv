"""Simple validation test to verify custom task system works.

Run this directly to check your contribution:
    source .venv/bin/activate
    python3 tests/envs/test_custom_validation.py
"""

import sys
import os

# Add project root to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_imports():
    """Verify all custom task modules can be imported."""
    print("Testing imports...")
    try:
        from envs.browsergym_env.server.custom.custom_base import (  # noqa: F401
            CustomBrowserGymEnvironment,
        )
        from envs.browsergym_env.server.custom.custom_models import (  # noqa: F401
            CustomGymAction,
            CustomGymObservation,
            CustomGymState,
        )
        from envs.browsergym_env.server.custom.custom_tasks import (  # noqa: F401
            _CUSTOM_TASKS,
            get_custom_task,
            register_custom_task,
        )

        print("  PASS: All imports successful")
        return True
    except ImportError as e:
        print(f"  FAIL: Import failed: {e}")
        return False


def test_task_registration():
    """Verify custom tasks are registered."""
    print("\nTesting task registration...")
    try:
        from envs.browsergym_env.server.custom.custom_tasks import (
            _CUSTOM_TASKS,
            get_custom_task,
        )

        print(f"  Registered tasks: {list(_CUSTOM_TASKS.keys())}")

        expected_tasks = ["copy-paste", "copy-paste-multitab"]
        for task_name in expected_tasks:
            if task_name in _CUSTOM_TASKS:
                print(f"  PASS: Task '{task_name}' registered")
            else:
                print(f"  FAIL: Task '{task_name}' NOT found")
                return False

        # Test retrieval (returns instance, not class)
        task_instance = get_custom_task("copy-paste")
        print(f"  PASS: Retrieved task instance: {task_instance.__class__.__name__}")
        return True

    except Exception as e:
        print(f"  FAIL: Registration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_task_interface():
    """Verify task classes implement required methods."""
    print("\nTesting task interface...")
    try:
        from envs.browsergym_env.server.custom.custom_tasks import get_custom_task

        task_instance = get_custom_task("copy-paste")
        required_methods = [
            "_get_task_url",
            "_get_goal_description",
            "_calculate_reward",
            "_check_done",
            "reset",
            "step",
            "close",
        ]

        all_present = True
        for method_name in required_methods:
            if hasattr(task_instance, method_name):
                print(f"  PASS: {method_name}()")
            else:
                print(f"  FAIL: {method_name}() MISSING")
                all_present = False

        return all_present

    except Exception as e:
        print(f"  FAIL: Interface test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_models():
    """Verify data models can be instantiated."""
    print("\nTesting model instantiation...")
    try:
        from envs.browsergym_env.server.custom.custom_models import (
            CustomGymAction,
            CustomGymObservation,
            CustomGymState,
        )

        # Test Action
        action = CustomGymAction(action_str="click('#button')")
        print(f"  PASS: CustomGymAction: {action.action_str}")

        # Test State
        state = CustomGymState(
            episode_id="test-123",
            step_count=5,
            benchmark="custom",
            task_name="copy-paste",
            max_steps=10,
        )
        print(f"  PASS: CustomGymState: episode={state.episode_id}, step={state.step_count}")

        # Test Observation
        obs = CustomGymObservation(
            text="Sample page",
            goal="Copy text from source to target",
            done=False,
            reward=0.5,
        )
        print(f"  PASS: CustomGymObservation: reward={obs.reward}")

        return True

    except Exception as e:
        print(f"  FAIL: Model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Custom BrowserGym Task System - Validation Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Task Registration", test_task_registration),
        ("Task Interface", test_task_interface),
        ("Model Instantiation", test_models),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("Custom task contribution is working correctly.")
    else:
        print("Some tests failed.")
        print("Please fix the issues above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
