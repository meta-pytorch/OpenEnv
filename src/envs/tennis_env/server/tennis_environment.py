# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tennis Environment Server Implementation.

This module wraps ALE's Tennis game and exposes it via the OpenEnv Environment interface.
"""

import uuid
from typing import Dict, Optional, Tuple

import numpy as np

from core.env_server import Action, Environment, Observation

from ..models import TennisAction, TennisObservation, TennisState

# Import ALE
try:
    from ale_py import ALEInterface, roms
except ImportError as e:
    raise ImportError(
        "ALE (Arcade Learning Environment) is not installed. "
        "Please install it with: pip install ale-py"
    ) from e


# Tennis action mapping (full 18-action space)
TENNIS_ACTION_MAPPING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE"
}


class TennisEnvironment(Environment):
    """
    Tennis Environment wrapper for OpenEnv.

    This environment wraps the Atari Tennis game via ALE and provides
    symbolic state extraction and reward shaping for RL training.

    Args:
        mode: Game mode (optional, typically 0 for standard tennis).
        difficulty: Game difficulty (optional, 0-3).
        repeat_action_probability: Sticky action probability (default 0.25 per ALE/Tennis-v5).
        frameskip: Number of frames to skip per action (default 4).
        render_mode: Rendering mode (not used, for compatibility).

    Example:
        >>> env = TennisEnvironment()
        >>> obs = env.reset()
        >>> print(obs.score)  # (0, 0)
        >>> obs = env.step(TennisAction(action_id=2, action_name="UP"))
        >>> print(obs.reward, obs.done)
    """

    def __init__(
        self,
        mode: Optional[int] = None,
        difficulty: Optional[int] = None,
        repeat_action_probability: float = 0.25,
        frameskip: int = 4,
        render_mode: Optional[str] = None,
        # Dynamic reward shaping parameters
        score_reward: float = 10.0,
        score_penalty: float = -5.0,
        rally_bonus_max: float = 1.0,
        rally_bonus_scale: float = 0.1,
        movement_bonus: float = 0.05,
        positioning_bonus: float = 0.1,
        center_bonus: float = 0.2,
    ):
        """
        Initialize Tennis environment with configurable reward shaping.

        Args:
            mode: Game mode (optional).
            difficulty: Game difficulty (0-3).
            repeat_action_probability: Sticky action probability (default 0.25).
            frameskip: Number of frames to skip per action (default 4).
            render_mode: Rendering mode (not used).
            score_reward: Reward bonus for scoring a point (default 10.0).
            score_penalty: Reward penalty for opponent scoring (default -5.0).
            rally_bonus_max: Maximum rally bonus reward (default 1.0).
            rally_bonus_scale: Scale factor for rally bonus (default 0.1).
            movement_bonus: Reward for active movement (default 0.05).
            positioning_bonus: Reward for good positioning (default 0.1).
            center_bonus: Reward for center court positioning (default 0.2).
        """
        super().__init__()

        self.mode = mode
        self.difficulty = difficulty
        self.repeat_action_probability = repeat_action_probability
        self.frameskip = frameskip
        self.render_mode = render_mode

        # Reward shaping parameters (configurable for dynamic RL)
        self.score_reward = score_reward
        self.score_penalty = score_penalty
        self.rally_bonus_max = rally_bonus_max
        self.rally_bonus_scale = rally_bonus_scale
        self.movement_bonus = movement_bonus
        self.positioning_bonus = positioning_bonus
        self.center_bonus = center_bonus

        # Create ALE interface
        self.ale = ALEInterface()

        # Configure ALE
        from ale_py import LoggerMode
        self.ale.setLoggerMode(LoggerMode.Error)  # Error mode only
        self.ale.setFloat("repeat_action_probability", repeat_action_probability)

        # Load Tennis ROM
        try:
            rom_path = roms.get_rom_path("tennis")
            self.ale.loadROM(rom_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load Tennis ROM: {e}\n"
                f"Make sure ale-py is properly installed with ROM support."
            ) from e

        # Set mode and difficulty if specified
        if mode is not None:
            self.ale.setMode(mode)
        if difficulty is not None:
            self.ale.setDifficulty(difficulty)

        # Get full action set (Tennis uses all 18 actions)
        self._action_set = self.ale.getLegalActionSet()

        # Get screen dimensions
        self.screen_height, self.screen_width = self.ale.getScreenDims()
        self.screen_shape = [self.screen_height, self.screen_width, 3]

        # Initialize state
        self._state = TennisState(
            previous_score=(0, 0),
            rally_length=0,
            total_points=0,
            agent_games_won=0,
            opponent_games_won=0,
        )

        # Track previous observation for reward shaping
        self._prev_obs: Optional[TennisObservation] = None

    def reset(self) -> Observation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial observation for the agent.
        """
        # Reset ALE
        self.ale.reset_game()

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0
        self._state.previous_score = (0, 0)
        self._state.rally_length = 0
        self._state.total_points = 0
        self._state.agent_games_won = 0
        self._state.opponent_games_won = 0

        # Get initial observation
        self._prev_obs = None
        obs = self._make_observation(base_reward=0.0)
        self._prev_obs = obs

        return obs

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: TennisAction containing the action_id to execute.

        Returns:
            Observation after action execution with shaped reward.

        Raises:
            ValueError: If action is not a TennisAction.
        """
        if not isinstance(action, TennisAction):
            raise ValueError(f"Expected TennisAction, got {type(action)}")

        # Validate action_id
        if action.action_id < 0 or action.action_id >= len(self._action_set):
            raise ValueError(
                f"Invalid action_id: {action.action_id}. "
                f"Valid range: [0, {len(self._action_set) - 1}]"
            )

        # Get actual ALE action
        ale_action = self._action_set[action.action_id]

        # Execute action with frameskip
        total_reward = 0.0
        for _ in range(self.frameskip):
            total_reward += self.ale.act(ale_action)
            if self.ale.game_over():
                break

        self._state.step_count += 1

        # Get observation with base reward
        obs = self._make_observation(base_reward=total_reward)

        # Apply reward shaping
        shaped_reward = self._shape_reward(total_reward, obs, self._prev_obs)
        obs.reward = shaped_reward

        # Update state tracking
        self._update_state(obs)

        # Store for next step
        self._prev_obs = obs

        return obs

    @property
    def state(self) -> TennisState:
        """Get current environment state."""
        return self._state

    def _make_observation(self, base_reward: float) -> TennisObservation:
        """
        Create a TennisObservation from current ALE state.

        Args:
            base_reward: Base reward from ALE.

        Returns:
            TennisObservation for the agent.
        """
        # Get RGB screen
        screen_rgb = self.ale.getScreenRGB()
        screen_flat = screen_rgb.flatten().tolist()

        # Extract symbolic features from screen
        symbolic_features = self._extract_symbolic_features(screen_rgb)

        # Get game info
        lives = self.ale.lives()
        episode_frame_number = self.ale.getEpisodeFrameNumber()
        frame_number = self.ale.getFrameNumber()
        done = self.ale.game_over()

        # Extract score (approximate from RAM or track via rewards)
        score = self._estimate_score()

        # Create legal actions list (all 18 actions for tennis)
        legal_actions = list(range(len(self._action_set)))
        action_meanings = [TENNIS_ACTION_MAPPING.get(i, f"ACTION_{i}") for i in legal_actions]

        # Create observation
        obs = TennisObservation(
            screen_rgb=screen_flat,
            screen_shape=self.screen_shape,
            score=score,
            ball_side=symbolic_features["ball_side"],
            my_position=symbolic_features["my_position"],
            opponent_position=symbolic_features["opponent_position"],
            legal_actions=legal_actions,
            action_meanings=action_meanings,
            rally_length=self._state.rally_length,
            lives=lives,
            episode_frame_number=episode_frame_number,
            frame_number=frame_number,
            done=done,
            reward=base_reward,
            metadata={
                "game_name": "tennis",
                "action_set": [str(a) for a in self._action_set],
            },
        )

        return obs

    def _extract_symbolic_features(self, screen_rgb: np.ndarray) -> Dict[str, str]:
        """
        Extract symbolic features from RGB screen for LLM training.

        This provides simplified game state that's easier for LLMs to reason about.

        Args:
            screen_rgb: RGB screen array (210, 160, 3).

        Returns:
            Dictionary with ball_side, my_position, opponent_position.
        """
        try:
            # Initialize defaults
            features = {
                "ball_side": "unknown",
                "my_position": "unknown",
                "opponent_position": "unknown"
            }

            # Convert to numpy if needed
            if not isinstance(screen_rgb, np.ndarray):
                screen_rgb = np.array(screen_rgb)

            # Ball detection: Look for bright white pixels (tennis ball)
            # Ball typically appears as bright pixels (high values in all channels)
            brightness = screen_rgb.mean(axis=2)
            bright_threshold = brightness.max() * 0.9 if brightness.max() > 0 else 255

            bright_pixels = np.where(brightness >= bright_threshold)
            if len(bright_pixels[0]) > 0:
                # Get centroid of bright pixels
                ball_y = int(np.mean(bright_pixels[0]))
                ball_x = int(np.mean(bright_pixels[1]))

                # Classify ball position (left/center/right)
                if ball_x < self.screen_width / 3:
                    features["ball_side"] = "left"
                elif ball_x > 2 * self.screen_width / 3:
                    features["ball_side"] = "right"
                else:
                    features["ball_side"] = "center"

            # Player detection: Look for orange pixels (agent) and blue pixels (opponent)
            # Orange player: High R, medium G, low B
            # Blue player: Low R, low G, high B

            # Orange detection (agent)
            orange_mask = (screen_rgb[:, :, 0] > 150) & (screen_rgb[:, :, 1] < 150) & (screen_rgb[:, :, 2] < 100)
            orange_pixels = np.where(orange_mask)
            if len(orange_pixels[0]) > 5:  # Need enough pixels to be confident
                agent_y = int(np.mean(orange_pixels[0]))
                # Classify position (top/middle/bottom)
                if agent_y < self.screen_height / 3:
                    features["my_position"] = "top"
                elif agent_y > 2 * self.screen_height / 3:
                    features["my_position"] = "bottom"
                else:
                    features["my_position"] = "middle"

            # Blue detection (opponent)
            blue_mask = (screen_rgb[:, :, 0] < 100) & (screen_rgb[:, :, 1] < 150) & (screen_rgb[:, :, 2] > 150)
            blue_pixels = np.where(blue_mask)
            if len(blue_pixels[0]) > 5:
                opponent_y = int(np.mean(blue_pixels[0]))
                if opponent_y < self.screen_height / 3:
                    features["opponent_position"] = "top"
                elif opponent_y > 2 * self.screen_height / 3:
                    features["opponent_position"] = "bottom"
                else:
                    features["opponent_position"] = "middle"

            return features

        except Exception as e:
            # If extraction fails, return defaults
            return {
                "ball_side": "unknown",
                "my_position": "unknown",
                "opponent_position": "unknown"
            }

    def _estimate_score(self) -> Tuple[int, int]:
        """
        Estimate current game score.

        Tennis scoring is complex, so we track points scored via rewards.
        This is an approximation and may not reflect exact tennis score.

        Returns:
            Tuple of (agent_score, opponent_score).
        """
        # For now, return tracked scores
        # In real tennis, scoring is: 0, 15, 30, 40, game
        # We simplify by tracking points won
        agent_score = self._state.agent_games_won
        opponent_score = self._state.opponent_games_won

        return (agent_score, opponent_score)

    def _shape_reward(
        self,
        base_reward: float,
        obs: TennisObservation,
        prev_obs: Optional[TennisObservation]
    ) -> float:
        """
        Apply configurable reward shaping to enhance learning signal.

        Uses dynamic reward parameters set during initialization, allowing
        experiments with different reward structures for RL training.

        Args:
            base_reward: Raw reward from ALE.
            obs: Current observation.
            prev_obs: Previous observation.

        Returns:
            Shaped reward value.
        """
        reward = base_reward

        # Detect scoring events (base_reward != 0 means point scored)
        if base_reward > 0:
            # Agent scored a point
            reward += self.score_reward
        elif base_reward < 0:
            # Opponent scored a point
            reward += self.score_penalty

        # Rally length bonus (reward keeping the ball in play)
        if prev_obs is not None and not obs.done:
            # If no point was scored, increase rally
            if abs(base_reward) < 0.01:  # No scoring event
                rally_bonus = min(
                    self.rally_bonus_scale * self._state.rally_length / 10.0,
                    self.rally_bonus_max
                )
                reward += rally_bonus

        # Movement bonus: Encourage active play
        if prev_obs is not None:
            # Check if position changed
            if obs.my_position != prev_obs.my_position and obs.my_position != "unknown":
                reward += self.movement_bonus

        # Positioning bonus: Reward moving toward ball
        if prev_obs is not None and obs.ball_side != "unknown":
            # Simplified: reward being on same side as ball
            if obs.ball_side == "left" and obs.my_position in ["top", "middle"]:
                reward += self.positioning_bonus
            elif obs.ball_side == "right" and obs.my_position in ["middle", "bottom"]:
                reward += self.positioning_bonus
            elif obs.ball_side == "center" and obs.my_position == "middle":
                reward += self.center_bonus

        return reward

    def _update_state(self, obs: TennisObservation) -> None:
        """
        Update internal state tracking based on observation.

        Args:
            obs: Current observation.
        """
        # Detect if point was scored
        if obs.score != self._state.previous_score:
            # Point was scored, reset rally
            self._state.rally_length = 0
            self._state.total_points += 1

            # Update game wins
            agent_score, opponent_score = obs.score
            self._state.agent_games_won = agent_score
            self._state.opponent_games_won = opponent_score

            # Update previous score
            self._state.previous_score = obs.score
        else:
            # Rally continues
            self._state.rally_length += 1
