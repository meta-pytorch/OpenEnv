# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generic Gymnasium environment server implementation."""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, Optional
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    raise ValueError("Please install gymnasium with: pip install gymnasium")
from gymnasium import spaces
import numpy.typing as npt
from core.env_server import Environment

from ..models import GymAction, GymObservation, GymState

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class GymnasiumEnvironment(Environment):
    """
    Generic Gymnasium environment wrapper for OpenEnv.

    Any Gymnasium environment can be served by providing its environment id.
    The wrapper handles common concerns such as seed management, type conversion,
    and JSON-friendly serialization of observations.
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.env_id = env_id
        self.render_mode = render_mode
        self.max_steps = max_steps if max_steps and max_steps > 0 else None
        self._initial_seed = seed
        self._next_seed = seed

        logger.info(
            "Creating Gymnasium environment '%s' (render_mode=%s, max_steps=%s, seed=%s)",
            env_id,
            render_mode,
            self.max_steps,
            seed,
        )

        self.env = gym.make(env_id, render_mode=render_mode)

        if self.max_steps is not None:
            self.env = gym.wrappers.TimeLimit(
                self.env, max_episode_steps=self.max_steps
            )

        self._action_space_metadata = self._describe_space(self.env.action_space)
        self._observation_space_metadata = self._describe_space(
            self.env.observation_space
        )
        self._legal_actions = self._summarize_action_space(self.env.action_space)

        self._state = GymState(
            env_id=env_id,
            render_mode=render_mode,
            max_steps=self.max_steps,
            seed=seed,
        )

        logger.info("GymnasiumEnvironment for '%s' initialized", env_id)

    def reset(self) -> GymObservation:
        """Reset the environment and return the initial observation."""
        seed = self._consume_seed()
        obs, info = self.env.reset(seed=seed)

        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.episode_length = 0
        self._state.total_reward = 0.0
        self._state.seed = seed

        observation = self._make_observation(
            obs=obs,
            reward=None,
            done=False,
            info=info,
            terminated=False,
            truncated=False,
            raw_reward=0.0,
        )

        logger.info(
            "Environment '%s' reset (episode_id=%s, seed=%s)",
            self.env_id,
            self._state.episode_id,
            seed,
        )

        return observation

    def step(self, action: GymAction) -> GymObservation:
        """Execute an action and return the resulting observation."""
        gym_action = self._convert_action(action)
        obs, reward, terminated, truncated, info = self.env.step(gym_action)
        if action.return_frame:
            frame = self.env.render()
        else:
            frame = None

        self._state.step_count += 1
        self._state.episode_length += 1

        reward_value, raw_reward = self._normalize_reward(reward)
        if reward_value is not None:
            self._state.total_reward += reward_value

        done = bool(terminated or truncated)

        observation = self._make_observation(
            obs=obs,
            reward=reward_value,
            done=done,
            info=info,
            terminated=terminated,
            truncated=truncated,
            raw_reward=raw_reward,
            frame=frame,
        )

        logger.debug(
            "Step %s -> reward=%s terminated=%s truncated=%s",
            self._state.step_count,
            reward,
            terminated,
            truncated,
        )

        return observation

    @property
    def state(self) -> GymState:
        """Return the current environment state."""
        return self._state

    def close(self) -> None:
        """Close the underlying Gymnasium environment."""
        logger.info("Closing GymnasiumEnvironment for '%s'", self.env_id)
        if hasattr(self.env, "close"):
            self.env.close()
        logger.info("GymnasiumEnvironment closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _consume_seed(self) -> Optional[int]:
        if self._next_seed is None:
            return None
        seed = self._next_seed
        self._next_seed += 1
        return seed

    def _convert_action(self, action: GymAction) -> Any:
        if not isinstance(action, GymAction):
            raise ValueError(f"Expected GymAction, received {type(action)}")

        raw_action = action.action
        space = self.env.action_space

        if space.contains(raw_action):
            return raw_action

        converted = self._convert_action_for_space(space, raw_action)

        if not space.contains(converted):
            raise ValueError(
                f"Action {raw_action!r} could not be converted for space {space}"
            )

        return converted

    def _convert_action_for_space(self, space: spaces.Space, value: Any) -> Any:
        if isinstance(space, spaces.Discrete):
            return int(value)

        if isinstance(space, spaces.MultiDiscrete):
            return np.asarray(value, dtype=space.dtype)

        if isinstance(space, spaces.MultiBinary):
            return np.asarray(value, dtype=space.dtype)

        if isinstance(space, spaces.Box):
            return np.asarray(value, dtype=space.dtype)

        if isinstance(space, spaces.Tuple):
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"Tuple action space expects list/tuple, received {type(value)}"
                )
            if len(value) != len(space.spaces):
                raise ValueError(
                    f"Tuple action with length {len(value)} does not match "
                    f"expected length {len(space.spaces)}"
                )
            return tuple(
                self._convert_action_for_space(subspace, subvalue)
                for subspace, subvalue in zip(space.spaces, value)
            )

        if isinstance(space, spaces.Dict):
            if not isinstance(value, dict):
                raise TypeError(
                    f"Dict action space expects dict, received {type(value)}"
                )
            return {
                key: self._convert_action_for_space(space.spaces[key], value[key])
                for key in space.spaces
            }

        if isinstance(space, spaces.Text):
            return str(value)

        return value

    def _normalize_reward(self, reward: Any) -> tuple[Optional[float], Any]:
        if isinstance(reward, (int, float)):
            value = float(reward)
            return value, value

        if isinstance(reward, (np.integer, np.floating)):
            value = float(reward.item())
            return value, value

        return None, self._to_serializable(reward)

    def _make_observation(
        self,
        obs: Any,
        reward: Optional[float],
        done: bool,
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
        raw_reward: Any,
        frame: Optional[npt.ArrayLike] = None,
    ) -> GymObservation:
        metadata = {
            "env_id": self.env_id,
            "render_mode": self.render_mode,
            "max_steps": self.max_steps,
            "seed": self._state.seed,
            "info": self._to_serializable(info),
            "raw_reward": raw_reward,
            "terminated": terminated,
            "truncated": truncated,
            "action_space": self._action_space_metadata,
            "observation_space": self._observation_space_metadata,
        }

        # Remove keys with None values for cleaner payloads
        metadata = {key: value for key, value in metadata.items() if value is not None}

        return GymObservation(
            state=self._to_serializable(obs),
            legal_actions=self._legal_actions,
            episode_length=self._state.episode_length,
            total_reward=self._state.total_reward,
            done=done,
            reward=reward,
            metadata=metadata,
            frame=self._to_serializable(frame),
        )

    def _describe_space(self, space: spaces.Space) -> Dict[str, Any]:
        description: Dict[str, Any] = {"type": type(space).__name__}

        if hasattr(space, "shape"):
            description["shape"] = self._to_serializable(getattr(space, "shape"))

        dtype = getattr(space, "dtype", None)
        if dtype is not None:
            description["dtype"] = str(dtype)

        if isinstance(space, spaces.Discrete):
            description["n"] = int(space.n)

        elif isinstance(space, spaces.MultiDiscrete):
            description["nvec"] = self._to_serializable(space.nvec)

        elif isinstance(space, spaces.MultiBinary):
            description["n"] = self._to_serializable(space.n)

        elif isinstance(space, spaces.Box):
            description["low"] = self._to_serializable(space.low)
            description["high"] = self._to_serializable(space.high)

        elif isinstance(space, spaces.Tuple):
            description["spaces"] = [
                self._describe_space(subspace) for subspace in space.spaces
            ]

        elif isinstance(space, spaces.Dict):
            description["spaces"] = {
                key: self._describe_space(subspace)
                for key, subspace in space.spaces.items()
            }

        elif isinstance(space, spaces.Text):
            description["min_length"] = space.min_length
            description["max_length"] = space.max_length

        return description

    def _summarize_action_space(self, space: spaces.Space) -> Any:
        if isinstance(space, spaces.Discrete):
            return list(range(int(space.n)))

        if isinstance(space, spaces.MultiDiscrete):
            return [list(range(int(n))) for n in self._to_serializable(space.nvec)]

        if isinstance(space, spaces.MultiBinary):
            return [0, 1]

        if isinstance(space, spaces.Box):
            return {
                "low": self._to_serializable(space.low),
                "high": self._to_serializable(space.high),
            }

        if isinstance(space, spaces.Tuple):
            return [self._summarize_action_space(subspace) for subspace in space.spaces]

        if isinstance(space, spaces.Dict):
            return {
                key: self._summarize_action_space(subspace)
                for key, subspace in space.spaces.items()
            }

        if isinstance(space, spaces.Text):
            return {"charset": "unicode"}

        return None

    def _to_serializable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return [self._to_serializable(v) for v in value.tolist()]

        if isinstance(value, (np.floating, np.integer)):
            return self._to_serializable(value.item())

        if isinstance(value, np.bool_):
            return bool(value)

        if isinstance(value, (list, tuple, set)):
            return [self._to_serializable(v) for v in value]

        if isinstance(value, dict):
            return {str(k): self._to_serializable(v) for k, v in value.items()}

        if isinstance(value, (int, bool, float)) or value is None:
            return value

        return str(value)
