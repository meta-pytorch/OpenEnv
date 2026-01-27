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
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False


def test_task_registration():
    """Verify custom tasks are registered."""
    try:
        from envs.browsergym_env.server.custom.custom_tasks import (
            _CUSTOM_TASKS,
            get_custom_task,
        )

        expected_tasks = ["copy-paste", "copy-paste-multitab"]
        for task_name in expected_tasks:
            if task_name not in _CUSTOM_TASKS:
                print(f"Task '{task_name}' not registered")
                return False

        get_custom_task("copy-paste")
        return True

    except Exception as e:
        print(f"Registration test failed: {e}")
        return False


def test_task_interface():
    """Verify task classes implement required methods."""
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

        for method_name in required_methods:
            if not hasattr(task_instance, method_name):
                print(f"  Missing method: {method_name}")
                return False

        return True

    except Exception as e:
        print(f"  Interface test failed: {e}")
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
    print("Custom BrowserGym Task System - Validation Tests\n")

    tests = [
        ("Imports", test_imports),
        ("Task Registration", test_task_registration),
        ("Task Interface", test_task_interface),
        ("Model Instantiation", test_models),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            results.append((test_name, test_func()))
        except Exception as e:
            print(f"{test_name} crashed: {e}")
            results.append((test_name, False))

    print("\nResults:")
    all_passed = True
    for test_name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ All tests passed")
    else:
        print("\n❌ Some tests failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
