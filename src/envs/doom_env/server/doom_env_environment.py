# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Doom Environment Implementation.

Wraps ViZDoom for reinforcement learning research with OpenEnv interface.
ViZDoom is a Doom-based AI research platform for visual RL.
"""

import uuid
from typing import List, Literal, Optional

from models import DoomAction, DoomObservation

from openenv_core.env_server.interfaces import Environment
from openenv_core.env_server.types import State

# Import ViZDoom
try:
    import numpy as np
    import vizdoom as vzd
except ImportError as e:
    raise ImportError(
        "ViZDoom is not installed. " "Please install it with: pip install vizdoom"
    ) from e


class DoomEnvironment(Environment):
    """
    Doom Environment wrapper for OpenEnv.

    This environment wraps ViZDoom scenarios and provides a clean interface
    for RL training with visual observations and game variables.

    Args:
        scenario: Name of the scenario to load (e.g., "basic", "deadly_corridor").
                  Can also be a path to a .cfg file.
        screen_resolution: Screen resolution - one of ViZDoom's resolutions.
        screen_format: Screen format - "CRCGCB", "RGB24", "GRAY8", etc.
        window_visible: Whether to show the game window.
        use_discrete_actions: If True, use pre-defined discrete action space.
                             If False, use continuous button combinations.

    Example:
        >>> env = DoomEnvironment("basic")
        >>> obs = env.reset()
        >>> print(obs.screen_shape)  # e.g., [120, 160, 3]
        >>> obs = env.step(DoomAction(action_id=0))  # Take action
        >>> print(obs.reward, obs.done)
    """

    def __init__(
        self,
        scenario: str = "basic",
        screen_resolution: str = "RES_160X120",
        screen_format: str = "RGB24",
        window_visible: bool = False,
        use_discrete_actions: bool = True,
    ):
        """Initialize Doom environment."""
        super().__init__()

        self.scenario = scenario
        self.screen_resolution = screen_resolution
        self.screen_format = screen_format
        self.window_visible = window_visible
        self.use_discrete_actions = use_discrete_actions

        # Create DoomGame instance
        self.game = vzd.DoomGame()

        # Load configuration
        self._load_scenario(scenario)

        # Configure visual settings
        self._configure_visuals(screen_resolution, screen_format, window_visible)

        # Initialize the game
        self.game.init()

        # Get available actions
        self.available_buttons = self.game.get_available_buttons()
        self.num_buttons = len(self.available_buttons)

        # Create discrete action space if requested
        if use_discrete_actions:
            self._create_discrete_actions()
        else:
            self.discrete_actions = None

        # Get screen dimensions
        self._update_screen_shape()

        # Initialize state
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

        # Rendering state
        self._render_window = None

    def _load_scenario(self, scenario: str) -> None:
        """Load scenario configuration."""
        # Check if scenario is a path to .cfg file
        if scenario.endswith(".cfg"):
            self.game.load_config(scenario)
        else:
            # Try to load built-in scenario
            try:
                # ViZDoom scenarios are typically in scenarios/ directory
                scenario_path = vzd.scenarios_path + f"/{scenario}.cfg"
                self.game.load_config(scenario_path)
            except Exception:
                # If built-in scenario not found, try as direct path
                try:
                    self.game.load_config(scenario)
                except Exception as e:
                    raise ValueError(
                        f"Could not load scenario '{scenario}'. "
                        f"Provide either a built-in scenario name (e.g., 'basic') "
                        f"or a path to a .cfg file."
                    ) from e

    def _configure_visuals(
        self, screen_resolution: str, screen_format: str, window_visible: bool
    ) -> None:
        """Configure visual settings."""
        # Set screen resolution
        resolution_map = {
            "RES_160X120": vzd.ScreenResolution.RES_160X120,
            "RES_320X240": vzd.ScreenResolution.RES_320X240,
            "RES_640X480": vzd.ScreenResolution.RES_640X480,
            "RES_800X600": vzd.ScreenResolution.RES_800X600,
            "RES_1024X768": vzd.ScreenResolution.RES_1024X768,
        }
        if screen_resolution in resolution_map:
            self.game.set_screen_resolution(resolution_map[screen_resolution])
        else:
            # Try to use it directly as a ViZDoom resolution
            try:
                self.game.set_screen_resolution(
                    getattr(vzd.ScreenResolution, screen_resolution)
                )
            except AttributeError:
                raise ValueError(f"Invalid screen resolution: {screen_resolution}")

        # Set screen format
        format_map = {
            "RGB24": vzd.ScreenFormat.RGB24,
            "GRAY8": vzd.ScreenFormat.GRAY8,
            "CRCGCB": vzd.ScreenFormat.CRCGCB,
            "CBCGCR": vzd.ScreenFormat.CBCGCR,
            "DOOM_256_COLORS8": vzd.ScreenFormat.DOOM_256_COLORS8,
        }
        if screen_format in format_map:
            self.game.set_screen_format(format_map[screen_format])
        else:
            try:
                self.game.set_screen_format(getattr(vzd.ScreenFormat, screen_format))
            except AttributeError:
                raise ValueError(f"Invalid screen format: {screen_format}")

        # Set window visibility
        self.game.set_window_visible(window_visible)

    def _create_discrete_actions(self) -> None:
        """
        Create a discrete action space.

        Common actions for most scenarios:
        0: No action
        1: Move left
        2: Move right
        3: Move forward
        4: Move backward
        5: Turn left
        6: Turn right
        7: Attack/Shoot
        """
        # Create some basic discrete actions
        # Each action is a list of button presses
        self.discrete_actions = []

        # Action 0: No-op
        self.discrete_actions.append([0] * self.num_buttons)

        # Create single-button actions
        for i in range(self.num_buttons):
            action = [0] * self.num_buttons
            action[i] = 1
            self.discrete_actions.append(action)

        # You can add more complex combinations here if needed
        # For example, move forward + shoot:
        # if self.num_buttons >= 2:
        #     action = [0] * self.num_buttons
        #     action[0] = 1  # e.g., move forward
        #     action[1] = 1  # e.g., shoot
        #     self.discrete_actions.append(action)

    def _update_screen_shape(self) -> None:
        """Update screen shape based on current state."""
        if self.game.is_episode_finished():
            # Use default shape if episode is finished
            if self.screen_format == "RGB24":
                channels = 3
            elif self.screen_format == "GRAY8":
                channels = 1
            else:
                channels = 3  # default

            # Get resolution from settings
            res_str = self.screen_resolution.replace("RES_", "").replace("X", "x")
            width, height = map(int, res_str.lower().split("x"))
            self.screen_shape = (
                [height, width, channels] if channels > 1 else [height, width]
            )
        else:
            state = self.game.get_state()
            if state is not None and state.screen_buffer is not None:
                self.screen_shape = list(state.screen_buffer.shape)
            else:
                # Fallback
                self.screen_shape = [120, 160, 3]

    def reset(self) -> DoomObservation:
        """
        Reset the environment and return initial observation.

        Returns:
            Initial observation for the agent.
        """
        # Start new episode
        self.game.new_episode()

        # Reset state tracking
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0

        # Get initial observation
        return self._make_observation()

    def step(self, action: DoomAction) -> DoomObservation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: DoomAction containing either action_id or buttons.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is invalid.
        """
        if not isinstance(action, DoomAction):
            raise ValueError(f"Expected DoomAction, got {type(action)}")

        # Convert action to button presses
        if action.action_id is not None:
            # Use discrete action
            if not self.use_discrete_actions or self.discrete_actions is None:
                raise ValueError("discrete actions not enabled")
            if action.action_id < 0 or action.action_id >= len(self.discrete_actions):
                raise ValueError(
                    f"Invalid action_id: {action.action_id}. "
                    f"Valid range: [0, {len(self.discrete_actions) - 1}]"
                )
            buttons = self.discrete_actions[action.action_id]
        elif action.buttons is not None:
            # Use button combination
            if len(action.buttons) != self.num_buttons:
                raise ValueError(
                    f"Invalid button count: {len(action.buttons)}. "
                    f"Expected {self.num_buttons} buttons."
                )
            buttons = action.buttons
        else:
            raise ValueError("Either action_id or buttons must be provided")

        # Execute action
        reward = self.game.make_action(buttons)

        self._state.step_count += 1

        # Get observation
        obs = self._make_observation()
        obs.reward = float(reward)

        return obs

    @property
    def state(self) -> State:
        """Get current environment state."""
        return self._state

    def _make_observation(self) -> DoomObservation:
        """
        Create a DoomObservation from current game state.

        Returns:
            DoomObservation for the agent.
        """
        # Check if episode is finished
        episode_finished = self.game.is_episode_finished()

        if episode_finished:
            # Return empty observation when episode is done
            screen_flat = [0] * int(np.prod(self.screen_shape))
            game_vars = []
        else:
            # Get current state
            state = self.game.get_state()

            # Get screen buffer
            if state.screen_buffer is not None:
                screen = state.screen_buffer
                # Flatten screen for JSON serialization
                screen_flat = screen.flatten().tolist()
                self.screen_shape = list(screen.shape)
            else:
                screen_flat = [0] * int(np.prod(self.screen_shape))

            # Get game variables
            if hasattr(state, "game_variables"):
                game_vars = (
                    state.game_variables.tolist()
                    if state.game_variables is not None
                    else []
                )
            else:
                game_vars = []

        # Get available actions
        if self.use_discrete_actions and self.discrete_actions is not None:
            available_actions = list(range(len(self.discrete_actions)))
        else:
            available_actions = []

        # Create observation
        obs = DoomObservation(
            screen_buffer=screen_flat,
            screen_shape=self.screen_shape,
            game_variables=game_vars,
            available_actions=available_actions,
            episode_finished=episode_finished,
            done=episode_finished,
            reward=0.0,  # Will be filled in by step()
            metadata={
                "scenario": self.scenario,
                "num_buttons": self.num_buttons,
                "available_buttons": [str(b) for b in self.available_buttons],
            },
        )

        return obs

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.

        Args:
            mode: Render mode - "human" for window display, "rgb_array" for array return.

        Returns:
            RGB array if mode is "rgb_array", None otherwise.
        """
        if self.game.is_episode_finished():
            # Can't render if episode is finished
            if mode == "rgb_array":
                return np.zeros(self.screen_shape, dtype=np.uint8)
            return None

        state = self.game.get_state()
        if state is None or state.screen_buffer is None:
            if mode == "rgb_array":
                return np.zeros(self.screen_shape, dtype=np.uint8)
            return None

        screen = state.screen_buffer

        if mode == "rgb_array":
            # Return the screen buffer as numpy array
            return screen
        elif mode == "human":
            # Display using matplotlib or cv2
            try:
                import cv2

                # Create window if it doesn't exist
                if self._render_window is None:
                    self._render_window = "ViZDoom - Doom Environment"
                    cv2.namedWindow(self._render_window, cv2.WINDOW_NORMAL)

                # Convert to BGR for OpenCV (if RGB)
                if len(screen.shape) == 3 and screen.shape[2] == 3:
                    screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                else:
                    screen_bgr = screen

                # Display
                cv2.imshow(self._render_window, screen_bgr)
                cv2.waitKey(1)  # Small delay to update window

            except ImportError:
                # Fallback to matplotlib if cv2 not available
                try:
                    import matplotlib.pyplot as plt

                    if self._render_window is None:
                        plt.ion()  # Enable interactive mode
                        self._render_window = plt.figure(figsize=(8, 6))
                        self._render_window.canvas.manager.set_window_title(
                            "ViZDoom - Doom Environment"
                        )

                    plt.clf()
                    if len(screen.shape) == 3:
                        plt.imshow(screen)
                    else:
                        plt.imshow(screen, cmap="gray")
                    plt.axis("off")
                    plt.pause(0.001)  # Small delay to update window

                except ImportError:
                    print(
                        "Warning: Neither cv2 nor matplotlib available for rendering. "
                        "Install with: pip install opencv-python or pip install matplotlib"
                    )
            return None
        else:
            raise ValueError(
                f"Invalid render mode: {mode}. Use 'human' or 'rgb_array'."
            )

    def close(self) -> None:
        """Clean up resources."""
        # Close render window if it exists
        if self._render_window is not None:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ImportError:
                try:
                    import matplotlib.pyplot as plt

                    plt.close("all")
                except ImportError:
                    pass
            self._render_window = None

        # Close the game
        if hasattr(self, "game") and self.game is not None:
            self.game.close()

    def __del__(self):
        """Destructor to ensure game is closed."""
        self.close()
