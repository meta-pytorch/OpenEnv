"""BrowserGym Environment implementation for OpenEnv.

This module wraps the BrowserGym framework to provide a compatible interface
with OpenEnv's Environment ABC. BrowserGym includes multiple benchmarks:
- MiniWoB++: Training environment with 100+ simple web tasks
- WebArena: Realistic evaluation with 812 complex tasks
- VisualWebArena: Visual web navigation tasks
- WorkArena: Enterprise task automation
"""

import importlib
import os
import sys
from typing import Any, Dict, Optional, TYPE_CHECKING
from uuid import uuid4

import gymnasium as gym

from core.env_server.interfaces import Environment
from envs.browsergym_env.models import (
    BrowserGymAction,
    BrowserGymObservation,
    BrowserGymState,
)

# Add the server directory to sys.path to allow custom module imports
_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

# Import custom models for custom benchmark
# Use TYPE_CHECKING to avoid runtime import issues with type hints
if TYPE_CHECKING:
    from custom.custom_models import (
        CustomGymAction,
        CustomGymObservation,
        CustomGymState,
    )

try:
    from custom.custom_models import (
        CustomGymAction as _CustomGymAction,
        CustomGymObservation as _CustomGymObservation,
        CustomGymState as _CustomGymState,
    )
    CUSTOM_AVAILABLE = True
    CustomGymAction = _CustomGymAction
    CustomGymObservation = _CustomGymObservation
    CustomGymState = _CustomGymState
    _CUSTOM_IMPORT_ERROR = None
except ImportError as e:
    CUSTOM_AVAILABLE = False
    CustomGymAction = None  # type: ignore
    CustomGymObservation = None  # type: ignore
    CustomGymState = None  # type: ignore
    _CUSTOM_IMPORT_ERROR = str(e)



class BrowserGymEnvironment(Environment):
    """BrowserGym environment wrapper for OpenEnv.

    This environment wraps BrowserGym's Gymnasium-compatible environments to
    provide unified access to multiple web navigation benchmarks.
    """

    def __init__(
        self,
        benchmark: str = "miniwob",
        task_name: Optional[str] = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout: float = 10000.0,
        **gym_kwargs: Any,
    ):
        """Initialize the BrowserGym environment.

        Args:
            benchmark: Benchmark to use ('miniwob', 'webarena', 'visualwebarena', etc.)
            task_name: Specific task within the benchmark (e.g., 'click-test', 'click-button')
                      If None, will use first available task
            headless: Whether to run browser in headless mode
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            timeout: Action timeout in milliseconds
            **gym_kwargs: Additional arguments passed to gym.make()
        """
        super().__init__()

        self.benchmark = benchmark
        self.task_name = task_name
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout = timeout
        self.gym_kwargs = gym_kwargs

        # Check if this is a custom benchmark
        self.is_custom = benchmark == "custom"

        if self.is_custom:
            # Handle custom benchmark differently
            if not CUSTOM_AVAILABLE:
                raise ValueError(
                    f"Custom benchmark requested but custom models not available.\n"
                    f"Import error: {_CUSTOM_IMPORT_ERROR}\n"
                    f"Make sure custom/custom_models.py exists in {_SERVER_DIR}/custom/"
                )
            
            if not task_name:
                raise ValueError("task_name is required for custom benchmark")
            
            # Import and instantiate custom environment
            try:
                from custom.custom_tasks import get_custom_task
                self.custom_env = get_custom_task(
                    task_name=task_name,
                    headless=headless,
                    viewport_width=viewport_width,
                    viewport_height=viewport_height,
                    timeout=timeout,
                    **gym_kwargs
                )
            except ImportError as e:
                raise ValueError(
                    f"Failed to import custom task '{task_name}': {e}\n"
                    f"Make sure the task is registered in custom/custom_tasks.py"
                ) from e
            
            self.gym_env = None
            self.env_id = f"custom/{task_name}"
            
            # Use CustomGymState for custom benchmarks
            self._state = CustomGymState(
                episode_id=str(uuid4()),
                step_count=0,
                benchmark="custom",
                task_name=task_name,
            )
        else:
            # Original BrowserGym benchmark handling
            # Build environment ID
            if task_name:
                self.env_id = f"browsergym/{benchmark}.{task_name}"
            else:
                self.env_id = f"browsergym/{benchmark}"

            # force import the benchmark module
            benchmark_modules = {
                "miniwob": "browsergym.envs.miniwob",
                "webarena": "browsergym.envs.webarena",
                "visualwebarena": "browsergym.envs.visualwebarena",
                "workarena": "browsergym.envs.workarena",
            }
            module_path = benchmark_modules.get(benchmark)
            try:
                if module_path:
                    importlib.import_module(module_path)
                else:
                    importlib.import_module("browsergym")
            except ModuleNotFoundError as import_error:
                raise ValueError(
                    f"Failed to import BrowserGym benchmark '{benchmark}': {import_error}\n"
                    f"Make sure the package browsergym-{benchmark} is installed."
                ) from import_error

            # Create the BrowserGym environment
            try:
                self.gym_env = gym.make(
                    self.env_id,
                    headless=headless,
                    viewport={"width": viewport_width, "height": viewport_height},
                    timeout=timeout,
                    **gym_kwargs,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create BrowserGym environment '{self.env_id}': {e}\n"
                    f"Make sure the benchmark is installed (e.g., pip install browsergym-{benchmark})"
                ) from e

            # State tracking for standard benchmarks
            self._state = BrowserGymState(
                episode_id=str(uuid4()),
                step_count=0,
                benchmark=benchmark,
                task_name=task_name or "",
            )
            
            self.custom_env = None

        self._last_obs: Optional[Dict[str, Any]] = None
        self._last_info: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        task_name: Optional[str] = None,
    ) -> BrowserGymObservation:
        """Reset the environment with a specific task.

        Args:
            seed: Random seed for reproducibility
            task_name: Override task name for this episode

        Returns:
            Initial observation for the task
        """
        if self.is_custom:
            # Handle custom environment reset
            obs = self.custom_env.reset(seed=seed)
            self._state = self.custom_env.state
            # Convert CustomGymObservation to BrowserGymObservation
            return self._convert_custom_observation(obs)
        
        # Original BrowserGym handling
        # Generate new episode ID
        self._state = BrowserGymState(
            episode_id=str(uuid4()),
            step_count=0,
            benchmark=self.benchmark,
            task_name=task_name or self.task_name or "",
        )

        # Reset options
        reset_options = {}
        if seed is not None:
            reset_options["seed"] = seed

        # Reset the gym environment
        obs, info = self.gym_env.reset(**reset_options)

        self._last_obs = obs
        self._last_info = info

        # Extract observation details
        return self._create_observation(obs, info, done=False, reward=0.0)

    def step(self, action: BrowserGymAction) -> BrowserGymObservation:
        """Execute an action in the environment.

        Args:
            action: The action to execute

        Returns:
            Observation after executing the action
        """
        if self.is_custom:
            # Convert BrowserGymAction to CustomGymAction
            custom_action = CustomGymAction(action_str=action.action_str)
            obs = self.custom_env.step(custom_action)
            self._state = self.custom_env.state
            # Convert CustomGymObservation to BrowserGymObservation
            return self._convert_custom_observation(obs)
        
        # Original BrowserGym handling
        self._state.step_count += 1

        # Execute action in gym environment
        try:
            obs, reward, terminated, truncated, info = self.gym_env.step(
                action.action_str
            )

            self._last_obs = obs
            self._last_info = info

            # Update state
            done = terminated or truncated
            self._state.cum_reward += float(reward)

            # Extract goal from info if available
            if "goal" in info:
                self._state.goal = str(info["goal"])

            return self._create_observation(obs, info, done=done, reward=float(reward))

        except Exception as e:
            # Handle action execution errors
            error_msg = str(e)
            return BrowserGymObservation(
                text=self._last_obs.get("text", "") if self._last_obs else "",
                url=self._last_obs.get("url", "") if self._last_obs else "",
                goal=self._state.goal,
                error=error_msg,
                last_action_error=True,
                done=False,
                reward=0.0,
            )

    def _create_observation(
        self,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        done: bool,
        reward: float,
    ) -> BrowserGymObservation:
        """Convert BrowserGym observation to OpenEnv format.

        Args:
            obs: BrowserGym observation dict
            info: BrowserGym info dict
            done: Whether episode is done
            reward: Reward for the step

        Returns:
            BrowserGymObservation
        """
        # Extract text observation (could be AXTree, DOM, or other)
        text = ""
        if "axtree_txt" in obs:
            text = obs["axtree_txt"]
        elif "pruned_html" in obs:
            text = obs["pruned_html"]
        elif "dom_txt" in obs:
            text = obs["dom_txt"]
        elif isinstance(obs, str):
            text = obs

        # Extract URL
        url = info.get("url", "")
        if not url and "page" in info:
            url = info["page"].get("url", "")

        # Extract goal/instruction
        goal = info.get("goal", "")
        if not goal and "task" in info:
            goal = info["task"].get("goal", "")

        # Update state
        self._state.current_url = url
        self._state.goal = goal

        # Extract additional observation modalities
        screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
        axtree_txt = obs.get("axtree_txt", "") if isinstance(obs, dict) else ""
        pruned_html = obs.get("pruned_html", "") if isinstance(obs, dict) else ""

        # Store full BrowserGym observation and info in metadata
        # This preserves timestamps, additional fields, and any future extensions
        browsergym_metadata = {
            "browsergym_obs": obs if isinstance(obs, dict) else {},
            "browsergym_info": info,
        }

        return BrowserGymObservation(
            text=text,
            url=url,
            screenshot=screenshot,
            goal=goal,
            axtree_txt=axtree_txt,
            pruned_html=pruned_html,
            error="",
            last_action_error=False,
            done=done,
            reward=reward,
            metadata=browsergym_metadata,
        )

    def _convert_custom_observation(
        self, custom_obs: "CustomGymObservation"  # type: ignore
    ) -> BrowserGymObservation:
        """Convert CustomGymObservation to BrowserGymObservation.

        Args:
            custom_obs: Custom observation to convert

        Returns:
            BrowserGymObservation
        """
        return BrowserGymObservation(
            text=custom_obs.text,
            url=custom_obs.url,
            screenshot=custom_obs.screenshot,
            goal=custom_obs.goal,
            axtree_txt=custom_obs.axtree_txt,
            pruned_html=custom_obs.pruned_html,
            error=custom_obs.error,
            last_action_error=custom_obs.last_action_error,
            done=custom_obs.done,
            reward=custom_obs.reward,
            metadata={
                "custom_data": custom_obs.custom_data,
                **(custom_obs.metadata or {}),
            },
        )

    @property
    def state(self) -> BrowserGymState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up environment resources."""
        if self.is_custom:
            if self.custom_env:
                self.custom_env.close()
        else:
            if hasattr(self, "gym_env"):
                self.gym_env.close()
