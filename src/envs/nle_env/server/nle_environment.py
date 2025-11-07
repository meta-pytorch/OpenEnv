# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NetHack Learning Environment Implementation.

This module wraps the NLE (NetHack Learning Environment) as an OpenEnv
environment, providing HTTP-based access to NetHack for RL training.
"""

import time
from typing import Optional

from core.env_server.interfaces import Environment, Transform

from ..models import NLEAction, NLEObservation, NLEState

# Import NLE - will be installed in Docker
try:
    from nle.env import NLE
except ImportError:
    NLE = None  # type: ignore


class NLEEnvironment(Environment):
    """
    OpenEnv wrapper for the NetHack Learning Environment.

    This environment wraps NLE's gym interface and provides OpenEnv-compatible
    reset(), step(), and state access.

    With beefy compute, we use simple JSON serialization and include all
    observation types by default. No optimization needed - compute handles it.

    Example:
        >>> env = NLEEnvironment()
        >>> obs = env.reset()
        >>> print(obs.reward)  # 0.0
        >>>
        >>> obs = env.step(NLEAction(action_id=0))  # Move north
        >>> print(obs.reward)  # Score delta
        >>> print(env.state.step_count)  # 1
    """

    def __init__(
        self,
        task_name: str = "score",
        character: str = "mon-hum-neu-mal",
        max_episode_steps: int = 5000,
        observation_keys: tuple = (
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "inv_glyphs",
            "inv_strs",
            "inv_letters",
            "inv_oclasses",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
        ),
        transform: Optional[Transform] = None,
    ):
        """
        Initialize the NLE environment.

        Args:
            task_name: Task variant (score, staircase, oracle, gold, etc.)
            character: Character definition (role-race-gender-alignment)
            max_episode_steps: Maximum steps before episode is aborted
            observation_keys: Which observations to include
            transform: Optional observation transform
        """
        super().__init__(transform=transform)

        if NLE is None:
            raise ImportError(
                "NLE is not installed. Install with: pip install nle\n"
                "For Docker builds, this will be installed automatically."
            )

        self._task_name = task_name
        self._character = character
        self._observation_keys = observation_keys

        # Create NLE gym environment
        # With beefy compute: no ttyrec saving, all observations enabled
        self.nle_env = NLE(
            character=character,
            observation_keys=observation_keys,
            max_episode_steps=max_episode_steps,
            save_ttyrec_every=0,  # Disable by default (can enable via env var)
            wizard=False,  # Can enable via env var for debugging
            spawn_monsters=True,
        )

        # Episode tracking
        self._episode_id: Optional[str] = None
        self._step_count = 0
        self._last_reward = 0.0
        self._last_done = False
        self._end_status = "RUNNING"
        self._in_normal_game = False

    def reset(self) -> NLEObservation:
        """
        Reset the environment and return initial observation.

        Returns:
            NLEObservation with initial game state
        """
        # Reset NLE gym env
        # Note: Gym 0.26+ returns (obs, info) tuple from reset()
        reset_result = self.nle_env.reset()

        # Handle both old gym API (returns obs dict) and new API (returns tuple)
        if isinstance(reset_result, tuple):
            gym_obs, _ = reset_result  # Unpack (observation, info)
        else:
            gym_obs = reset_result  # Old API

        # Initialize episode tracking
        self._episode_id = f"nle_{int(time.time() * 1000000)}"
        self._step_count = 0
        self._last_reward = 0.0
        self._last_done = False
        self._end_status = "RUNNING"
        self._in_normal_game = self.nle_env.nethack.in_normal_game()

        # Convert gym observation to OpenEnv observation
        obs = self._convert_observation(gym_obs, reward=0.0, done=False)

        return self._apply_transform(obs)

    def step(self, action: NLEAction) -> NLEObservation:  # type: ignore[override]
        """
        Execute action in NetHack and return observation.

        Args:
            action: NLEAction with action_id (0-112)

        Returns:
            NLEObservation with game state after action
        """
        # Execute action in NLE
        # Note: Gym 0.26+ returns (obs, reward, terminated, truncated, info)
        #       Older gym returns (obs, reward, done, info)
        step_result = self.nle_env.step(action.action_id)

        # Handle both old and new gym APIs
        if len(step_result) == 5:
            # New gym API (0.26+): (obs, reward, terminated, truncated, info)
            gym_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        elif len(step_result) == 4:
            # Old gym API: (obs, reward, done, info)
            gym_obs, reward, done, info = step_result
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")

        # Update tracking
        self._step_count += 1
        self._last_reward = float(reward)
        self._last_done = bool(done)
        self._end_status = str(info.get("end_status", "RUNNING"))
        self._in_normal_game = self.nle_env.nethack.in_normal_game()

        # Convert observation
        obs = self._convert_observation(gym_obs, reward=reward, done=done)

        # Add metadata from NLE
        obs.metadata.update(
            {
                "end_status": self._end_status,
                "is_ascended": info.get("is_ascended", False),
            }
        )

        return self._apply_transform(obs)

    @property
    def state(self) -> NLEState:
        """
        Get current environment state.

        Returns:
            NLEState with episode and game information
        """
        return NLEState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            game_over=self._last_done,
            end_status=self._end_status,
            in_normal_game=self._in_normal_game,
            character=self._character,
            task_name=self._task_name,
        )

    def _convert_observation(
        self, gym_obs: dict, reward: float, done: bool
    ) -> NLEObservation:
        """
        Convert NLE gym observation to NLEObservation.

        With beefy compute, we just convert numpy arrays to lists.
        No compression, no optimization - simplicity first.

        Args:
            gym_obs: Dictionary from NLE gym env
            reward: Reward for this step
            done: Whether episode is done

        Returns:
            NLEObservation with serialized arrays
        """
        obs_dict = {
            "reward": float(reward),
            "done": bool(done),
            "metadata": {},
        }

        # Convert each observation type from numpy array to nested list
        # This is simple and works perfectly with JSON + beefy compute
        for key in self._observation_keys:
            if key in gym_obs:
                array = gym_obs[key]
                # Convert numpy array to nested list for JSON serialization
                obs_dict[key] = array.tolist()

        return NLEObservation(**obs_dict)

    def close(self):
        """Clean up NLE environment."""
        if hasattr(self, "nle_env"):
            self.nle_env.close()
