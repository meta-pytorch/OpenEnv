"""Tests for the generic Gymnasium environment integration."""

import sys
from pathlib import Path

import pytest


try:
    # Ensure gymnasium is available; skip the whole module if it's missing.
    import importlib

    if importlib.util.find_spec("gymnasium") is None:
        raise ModuleNotFoundError
except ModuleNotFoundError:
    pytest.skip("gymnasium not installed", allow_module_level=True)

from envs.gym_env.client import GymAction, GymEnvironment
from envs.gym_env.server.gymnasium_environment import GymnasiumEnvironment


ENV_ID = "BipedalWalker-v3"


@pytest.fixture(name="env")
def fixture_env():
    env = GymnasiumEnvironment(env_id=ENV_ID, seed=123, render_mode="rgb_array")
    yield env
    env.close()


def test_bipedalwalker_reset_and_step(env: GymnasiumEnvironment):
    """Reset and step the BipedalWalker environment (continuous actions).

    The BipedalWalker environment uses a continuous Box action space, so
    the test checks that there are no discrete `legal_actions` and that the
    reported action_space metadata describes a Box (with numeric low/high lists).
    """
    obs = env.reset()
    state = env.state

    assert state.env_id == ENV_ID
    assert state.step_count == 0
    # Continuous environments typically don't expose discrete legal_actions
    # (set to None or empty). Accept either case.
    assert obs.legal_actions == {
        "low": [-1.0, -1.0, -1.0, -1.0],
        "high": [1.0, 1.0, 1.0, 1.0],
    }
    assert isinstance(obs.state, list)

    # Provide a sample continuous action. The client/server should convert
    # python lists into the correct numeric action shape for Gym.
    # Use a small vector; the environment will validate internally.
    sample_action = [0.0, 0.0, 0.0, 0.0]
    next_obs = env.step(GymAction(action=sample_action))
    assert env.state.step_count == 1
    assert isinstance(next_obs.state, list)
    assert next_obs.reward is not None
    assert "action_space" in next_obs.metadata
    # Expect a Box action space for BipedalWalker
    assert next_obs.metadata["action_space"]["type"] in ("Box", "box")
    low = next_obs.metadata["action_space"].get("low")
    high = next_obs.metadata["action_space"].get("high")
    assert isinstance(low, list) and isinstance(high, list)
    assert len(low) == len(high)


def test_continuous_action_conversion_and_metadata():
    env = GymnasiumEnvironment(env_id="MountainCarContinuous-v0", seed=42)
    # Capture initial observation from reset (some envs return different shapes on reset)
    _ = env.reset()

    obs = env.step(GymAction(action=[0.5]))
    # State should be serializable to a list
    assert isinstance(obs.state, list)
    assert not isinstance(obs.state, tuple)

    # Action space metadata should describe a Box for continuous envs
    assert "action_space" in obs.metadata
    action_space = obs.metadata["action_space"]
    assert action_space["type"] in ("Box", "box")
    low = action_space["low"]
    high = action_space["high"]
    assert isinstance(low, list) and isinstance(high, list)
    assert len(low) == len(high) == 1

    env.close()


def test_lunarlander_environments():
    """Test both discrete and continuous versions of LunarLander.

    This test verifies that:
    1. Both discrete and continuous versions can be initialized
    2. Action spaces are correctly reported
    3. Observations and rewards are properly structured
    4. State transitions work as expected
    """
    # Test LunarLander-v2 (discrete actions)
    env_discrete = GymnasiumEnvironment(env_id="LunarLander-v3", seed=42)
    obs_discrete = env_discrete.reset()

    # Verify discrete action space
    assert obs_discrete.legal_actions == [0, 1, 2, 3]  # Four discrete actions
    assert isinstance(obs_discrete.state, list)
    assert len(obs_discrete.state) == 8  # LunarLander has 8 state components

    # Test a discrete action
    next_obs = env_discrete.step(GymAction(action=1))  # Main engine
    assert env_discrete.state.step_count == 1
    assert isinstance(next_obs.state, list)
    assert next_obs.reward is not None
    assert next_obs.metadata["action_space"]["type"] == "Discrete"
    env_discrete.close()

    # Test LunarLander-v2 with continuous actions
    env_continuous = GymnasiumEnvironment(
        env_id="LunarLander-v3", seed=42, continuous=True
    )
    obs_continuous = env_continuous.reset()

    # Verify continuous action space
    assert obs_continuous.legal_actions == {
        "low": [-1.0, -1.0],  # Main engine, left-right engines
        "high": [1.0, 1.0],
    }
    assert isinstance(obs_continuous.state, list)

    # Test a continuous action
    next_obs = env_continuous.step(GymAction(action=[0.5, 0.0]))
    assert env_continuous.state.step_count == 1
    assert isinstance(next_obs.state, list)
    assert next_obs.reward is not None
    assert next_obs.metadata["action_space"]["type"] in ("Box", "box")
    assert len(next_obs.metadata["action_space"]["low"]) == 2
    env_continuous.close()


def test_client_parsers_handle_payloads():
    client = GymEnvironment(base_url="http://localhost:9000")
    state = [
        0.0027464781887829304,
        6.556225798703963e-06,
        -0.0008549225749447942,
        -0.016000041738152504,
        0.09236064553260803,
        0.0019846635404974222,
        0.8599309325218201,
        -0.00017501995898783207,
        1.0,
        0.03271123394370079,
        0.001984562259167433,
        0.8535996675491333,
        -0.00135040411259979,
        1.0,
        0.4408135712146759,
        0.4458196759223938,
        0.461422324180603,
        0.4895496964454651,
        0.5341022610664368,
        0.6024604439735413,
        0.7091481685638428,
        0.8859308958053589,
        1.0,
        1.0,
    ]
    payload = {
        "observation": {
            "state": state,
            "legal_actions": {
                "low": [-1.0, -1.0, -1.0, -1.0],
                "high": [1.0, 1.0, 1.0, 1.0],
            },
            "episode_length": 0,
            "total_reward": 0.0,
            "metadata": {
                "env_id": "BipedalWalker-v3",
                "render_mode": "rgb_array",
                "seed": 124,
                "info": {},
                "raw_reward": 0.0,
                "terminated": False,
                "truncated": False,
                "action_space": {
                    "type": "Box",
                    "shape": [4],
                    "dtype": "float32",
                    "low": [-1.0, -1.0, -1.0, -1.0],
                    "high": [1.0, 1.0, 1.0, 1.0],
                },
                "observation_space": {
                    "type": "Box",
                    "shape": [24],
                    "dtype": "float32",
                    "low": [
                        -3.1415927410125732,
                        -5.0,
                        -5.0,
                        -5.0,
                        -3.1415927410125732,
                        -5.0,
                        -3.1415927410125732,
                        -5.0,
                        -0.0,
                        -3.1415927410125732,
                        -5.0,
                        -3.1415927410125732,
                        -5.0,
                        -0.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                    ],
                    "high": [
                        3.1415927410125732,
                        5.0,
                        5.0,
                        5.0,
                        3.1415927410125732,
                        5.0,
                        3.1415927410125732,
                        5.0,
                        5.0,
                        3.1415927410125732,
                        5.0,
                        3.1415927410125732,
                        5.0,
                        5.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ],
                },
            },
        },
        "reward": 0.0,
        "done": False,
    }

    result = client._parse_result(payload)
    assert result.observation.state == state
    assert result.observation.legal_actions == {
        "low": [-1.0, -1.0, -1.0, -1.0],
        "high": [1.0, 1.0, 1.0, 1.0],
    }
    assert result.reward == 0.0
    assert result.done is False

    client.close()


def test_cartpole_discrete_action_space_and_step():
    env = GymnasiumEnvironment(env_id="CartPole-v1", seed=7)
    obs = env.reset()

    # Discrete action space should expose 'n' in metadata and legal_actions as a list
    assert env.state.env_id == "CartPole-v1"
    assert "action_space" in obs.metadata
    action_meta = obs.metadata["action_space"]
    assert action_meta["type"] in ("Discrete", "discrete")
    assert "n" in action_meta and isinstance(action_meta["n"], int)

    # legal_actions should be a list of integers 0..n-1
    assert isinstance(obs.legal_actions, list)
    assert obs.legal_actions == list(range(action_meta["n"]))

    # Perform a step with a valid discrete action
    next_obs = env.step(GymAction(action=0))
    assert isinstance(next_obs.state, list) or next_obs.state is not None
    assert next_obs.reward is not None
    env.close()


def test_taxi_discrete_action_space():
    # Taxi is a classic discrete-action environment (n typically 6)
    env = GymnasiumEnvironment(env_id="Taxi-v3", seed=10)
    obs = env.reset()

    assert env.state.env_id == "Taxi-v3"
    assert "action_space" in obs.metadata
    action_meta = obs.metadata["action_space"]
    assert action_meta["type"] in ("Discrete", "discrete")
    assert action_meta.get("n", None) is not None
    assert isinstance(obs.legal_actions, list)

    # Try a valid action (0) and ensure step returns a serializable state
    next_obs = env.step(GymAction(action=0))
    assert next_obs.reward is not None
    assert next_obs.done in (True, False)
    env.close()


def test_pendulum_continuous_action_box():
    # Pendulum has a continuous Box action space of shape (1,)
    env = GymnasiumEnvironment(env_id="Pendulum-v1", seed=42)
    obs = env.reset()

    assert env.state.env_id == "Pendulum-v1"
    assert "action_space" in obs.metadata
    action_meta = obs.metadata["action_space"]
    assert action_meta["type"] in ("Box", "box")
    # Expect shape to be present and of length 1
    shape = action_meta.get("shape")
    assert isinstance(shape, list) or isinstance(shape, tuple)
    assert len(shape) >= 1

    # Provide a valid continuous action (single-element list)
    next_obs = env.step(GymAction(action=[0.0]))
    assert next_obs.reward is not None
    assert isinstance(next_obs.state, list) or next_obs.state is not None
    env.close()
