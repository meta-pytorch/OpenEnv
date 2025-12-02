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
from typing import Any, Dict, Optional
from uuid import uuid4

import gymnasium as gym

from openenv_core.env_server.interfaces import Environment
from browsergym_env.models import (
    BrowserGymAction,
    BrowserGymObservation,
    BrowserGymState,
)


_MINIWOB_LOAD_HELP = (
    "MiniWoB tasks require the MiniWoB HTML bundle to be served over HTTP. "
    "The official BrowserGym Docker image handles this automatically by "
    "serving the bundle on port 8888. For custom or non-Docker deployments, "
    "clone the MiniWoB++ repository, start a static server inside "
    "`miniwob-plusplus/miniwob/html` (e.g. `python -m http.server 8888`), and "
    "set the MINIWOB_URL environment variable to the served base URL such as "
    "`http://localhost:8888/miniwob/`."
)


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
        self.gym_kwargs = dict(gym_kwargs)

        # Build environment ID
        if task_name:
            self.env_id = f"browsergym/{benchmark}.{task_name}"
        else:
            self.env_id = f"browsergym/{benchmark}"

        # force import the benchmark module
        benchmark_modules = {
            "miniwob": "browsergym.miniwob",
            "webarena": "browsergym.webarena",
            "visualwebarena": "browsergym.visualwebarena",
            "workarena": "browsergym.workarena",
        }
        module_path = benchmark_modules.get(benchmark)
        try:
            if module_path:
                importlib.import_module(module_path)
            else:
                importlib.import_module("browsergym")
        except ModuleNotFoundError as import_error:
            message = (
                "Failed to import BrowserGym benchmark "
                f"'{benchmark}': {import_error}\n"
                "Install the matching browsergym package "
                f"(e.g., browsergym-{benchmark})."
            )
            raise ValueError(message) from import_error

        # Create the BrowserGym environment
        try:
            self.gym_env = gym.make(
                self.env_id,
                headless=headless,
                viewport={"width": viewport_width, "height": viewport_height},
                timeout=timeout,
                **self.gym_kwargs,
            )
        except Exception as e:  # noqa: BLE001 - gym.make
            message = (
                "Failed to create BrowserGym environment "
                f"'{self.env_id}': {e}\n"
                "Make sure the benchmark package is installed "
                f"(e.g., pip install browsergym-{benchmark})."
            )
            raise ValueError(message) from e

        # State tracking
        self._state = BrowserGymState(
            episode_id=str(uuid4()),
            step_count=0,
            benchmark=benchmark,
            task_name=task_name or "",
        )

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
        try:
            obs, info = self.gym_env.reset(**reset_options)
        except AttributeError as err:
            if "context" in str(err) and hasattr(self.gym_env, "close"):
                # BrowserGym can leave partially initialized state after a
                # failed reset. Close the hanging resources and try once more.
                self.gym_env.close()
                obs, info = self.gym_env.reset(**reset_options)
            else:
                raise
        except Exception as err:  # noqa: BLE001 - browsergym
            message = str(err)
            if self.benchmark == "miniwob" and "core is not defined" in message:
                raise ValueError(_MINIWOB_LOAD_HELP) from err
            raise

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

    @property
    def state(self) -> BrowserGymState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up environment resources."""
        if hasattr(self, "gym_env"):
            self.gym_env.close()
