"""Smoke tests for Android environment.

This test suite verifies that the Android environment modules can be imported and basic
functionality works without requiring a Docker container or running Android emulator.

Full integration tests with Docker and android_env will be added in a future PR.
"""

import pytest


def test_android_models_import():
    """Test that Android models can be imported."""
    from envs.android_env.models import AndroidAction, AndroidObservation

    # Create a simple action
    action = AndroidAction(
        tool_name="tap",
        parameters={"x": 0.5, "y": 0.5}
    )

    assert action.tool_name == "tap"
    assert action.parameters["x"] == 0.5
    assert action.parameters["y"] == 0.5


def test_android_action_all_types():
    """Test that all action types can be created."""
    from envs.android_env.models import AndroidAction

    action_types = [
        ("tap", {"x": 0.5, "y": 0.5}),
        ("swipe", {"x1": 0.5, "y1": 0.8, "x2": 0.5, "y2": 0.2}),
        ("long_press", {"x": 0.5, "y": 0.5}),
        ("double_tap", {"x": 0.5, "y": 0.5}),
        ("scroll_down", {"distance": 0.5}),
        ("scroll_up", {"distance": 0.5}),
        ("swipe_left", {"distance": 0.5}),
        ("swipe_right", {"distance": 0.5}),
        ("type_text", {"text": "Hello World"}),
        ("press_button", {"button": "HOME"}),
    ]

    for tool_name, parameters in action_types:
        action = AndroidAction(tool_name=tool_name, parameters=parameters)
        assert action.tool_name == tool_name
        assert action.parameters == parameters


def test_android_observation_structure():
    """Test that Android observations have correct structure."""
    from envs.android_env.models import AndroidObservation

    obs = AndroidObservation(
        screen_image="base64_encoded_image_data",
        screen_width=1080,
        screen_height=1920,
        timestamp_ms=1234567890,
        orientation=0,
        pixels_shape=(1920, 1080, 3),
        done=False,
        reward=0.0
    )

    assert obs.screen_width == 1080
    assert obs.screen_height == 1920
    assert obs.done is False
    assert obs.reward == 0.0


def test_gesture_builder_tap():
    """Test GestureBuilder tap primitive."""
    from envs.android_env.server.gestures import GestureBuilder

    actions = GestureBuilder.tap(0.5, 0.5)

    # Tap should be 2 primitives: TOUCH + LIFT
    assert len(actions) == 2
    assert actions[0]["action_type"] == 0  # TOUCH
    assert actions[1]["action_type"] == 1  # LIFT
    assert actions[0]["x"] == 0.5
    assert actions[0]["y"] == 0.5


def test_gesture_builder_swipe():
    """Test GestureBuilder swipe generates interpolated sequence."""
    from envs.android_env.server.gestures import GestureBuilder

    actions = GestureBuilder.swipe(0.0, 0.0, 1.0, 1.0, duration_ms=300, steps=10)

    # Swipe should have TOUCH + REPEATs + LIFT
    assert len(actions) > 2
    assert actions[0]["action_type"] == 0  # TOUCH at start
    assert actions[-1]["action_type"] == 1  # LIFT at end

    # Middle actions should be REPEAT
    for action in actions[1:-1]:
        assert action["action_type"] == 2  # REPEAT


def test_adb_commands_text_input():
    """Test ADB text input command generation."""
    from envs.android_env.server.gestures import ADBCommands

    # Simple text
    cmd = ADBCommands.text_input("Hello")
    assert "input text" in cmd
    assert "Hello" in cmd

    # Text with spaces (should be escaped)
    cmd = ADBCommands.text_input("Hello World")
    assert "input text" in cmd
    assert "%s" in cmd  # Spaces replaced with %s

    # Unicode text
    cmd = ADBCommands.text_input("世界 🌍")
    assert "input text" in cmd


def test_adb_commands_keyevent():
    """Test ADB keyevent command generation."""
    from envs.android_env.server.gestures import ADBCommands

    cmd = ADBCommands.keyevent(ADBCommands.KEYCODE_HOME)
    assert "input keyevent" in cmd
    assert "HOME" in cmd

    cmd = ADBCommands.keyevent(ADBCommands.KEYCODE_BACK)
    assert "input keyevent" in cmd
    assert "BACK" in cmd


def test_coordinate_clipping():
    """Test that GestureBuilder handles out-of-bounds coordinates gracefully."""
    from envs.android_env.server.gestures import GestureBuilder

    # Out of bounds coordinates should still generate valid gestures
    actions = GestureBuilder.tap(1.5, -0.5)
    assert len(actions) == 2
    assert actions[0]["action_type"] == 0  # TOUCH
    assert actions[1]["action_type"] == 1  # LIFT
    # Coordinates passed through (clipping happens in environment)
    assert isinstance(actions[0]["x"], (int, float))
    assert isinstance(actions[0]["y"], (int, float))


@pytest.mark.skipif(
    True,  # Always skip - requires Docker and android_env installed
    reason="Full integration tests require Docker with android_env. See src/envs/android_env/tests/"
)
def test_android_environment_full_integration():
    """Full integration test with actual environment.

    This test is skipped by default as it requires:
    - Docker with android_env installed
    - Android SDK and emulator
    - Task definition file
    - KVM support (Linux only)

    Run the full test suite with:
        cd src/envs/android_env/tests
        ./run_unit_tests.sh  # 63 unit tests (no dependencies)
        ./run_docker_tests.sh  # 42 integration tests (requires Docker)
    """
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
