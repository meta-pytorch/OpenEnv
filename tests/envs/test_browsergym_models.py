"""Unit tests for BrowserGym models."""

from envs.browsergym_env.models import (
    BrowserGymAction,
    BrowserGymObservation,
    BrowserGymState,
)


def test_browser_gym_action_creation():
    """Test creating a BrowserGymAction."""
    action = BrowserGymAction(action_str="click('button')")
    assert action.action_str == "click('button')"
    assert isinstance(action.metadata, dict)


def test_browser_gym_action_with_metadata():
    """Test creating a BrowserGymAction with metadata."""
    action = BrowserGymAction(
        action_str="fill('username', 'john')",
        metadata={"user": "test", "timestamp": 123456}
    )
    assert action.action_str == "fill('username', 'john')"
    assert action.metadata["user"] == "test"
    assert action.metadata["timestamp"] == 123456


def test_browser_gym_observation_creation():
    """Test creating a BrowserGymObservation."""
    obs = BrowserGymObservation(
        text="Sample page text",
        url="http://example.com",
        goal="Click the submit button",
        done=False,
        reward=0.5,
    )
    assert obs.text == "Sample page text"
    assert obs.url == "http://example.com"
    assert obs.goal == "Click the submit button"
    assert obs.done is False
    assert obs.reward == 0.5
    assert obs.error == ""
    assert obs.last_action_error is False


def test_browser_gym_observation_defaults():
    """Test BrowserGymObservation default values."""
    obs = BrowserGymObservation()
    assert obs.text == ""
    assert obs.url == ""
    assert obs.goal == ""
    assert obs.screenshot is None
    assert obs.axtree_txt == ""
    assert obs.pruned_html == ""
    assert obs.error == ""
    assert obs.last_action_error is False


def test_browser_gym_observation_with_error():
    """Test BrowserGymObservation with error."""
    obs = BrowserGymObservation(
        text="Error state",
        error="Element not found",
        last_action_error=True,
        done=False,
        reward=0.0,
    )
    assert obs.error == "Element not found"
    assert obs.last_action_error is True


def test_browser_gym_state_creation():
    """Test creating a BrowserGymState."""
    state = BrowserGymState(
        episode_id="test-episode-123",
        step_count=5,
        benchmark="miniwob",
        task_name="click-test",
        goal="Click the button",
        current_url="http://miniwob.com/click-test",
    )
    assert state.episode_id == "test-episode-123"
    assert state.step_count == 5
    assert state.benchmark == "miniwob"
    assert state.task_name == "click-test"
    assert state.goal == "Click the button"
    assert state.current_url == "http://miniwob.com/click-test"


def test_browser_gym_state_defaults():
    """Test BrowserGymState default values."""
    state = BrowserGymState()
    assert state.episode_id is None
    assert state.step_count == 0
    assert state.benchmark == ""
    assert state.task_name == ""
    assert state.task_id is None
    assert state.goal == ""
    assert state.current_url == ""
    assert state.max_steps is None
    assert state.cum_reward == 0.0


def test_browser_gym_state_with_webarena():
    """Test BrowserGymState for WebArena tasks."""
    state = BrowserGymState(
        episode_id="webarena-123",
        step_count=10,
        benchmark="webarena",
        task_name="0",
        task_id="shopping_001",
        goal="Find the cheapest laptop",
        current_url="http://shopping.com/products",
        max_steps=50,
        cum_reward=0.5,
    )
    assert state.benchmark == "webarena"
    assert state.task_name == "0"
    assert state.task_id == "shopping_001"
    assert state.max_steps == 50
    assert state.cum_reward == 0.5


def test_observation_with_all_modalities():
    """Test BrowserGymObservation with all observation types."""
    obs = BrowserGymObservation(
        text="Main text",
        url="http://example.com",
        screenshot=[[[255, 0, 0]]],  # Simple 1x1 red pixel
        goal="Test goal",
        axtree_txt="[1] RootWebArea",
        pruned_html="<html><body></body></html>",
        done=True,
        reward=1.0,
    )
    assert obs.text == "Main text"
    assert obs.screenshot == [[[255, 0, 0]]]
    assert obs.axtree_txt == "[1] RootWebArea"
    assert obs.pruned_html == "<html><body></body></html>"
