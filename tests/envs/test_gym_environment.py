"""Tests for the generic Gymnasium environment integration."""

import sys
from pathlib import Path

import pytest


try:
    pass
except ModuleNotFoundError:
    pytest.skip("gymnasium not installed", allow_module_level=True)

from envs.gym_environment.client import GymAction, GymEnvironment
from envs.gym_environment.server.gymnasium_environment import GymnasiumEnvironment


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
