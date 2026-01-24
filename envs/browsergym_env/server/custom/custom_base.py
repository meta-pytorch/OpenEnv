"""Base custom environment for BrowserGym custom tasks.

This module provides a base class for creating custom BrowserGym tasks that
are not part of the official benchmarks. It simulates the BrowserGym gym
environment interface using Playwright directly.
"""

import asyncio
from abc import abstractmethod
from typing import Any, Dict, Optional
from uuid import uuid4

from playwright.async_api import async_playwright, Browser, Page, Playwright

from .custom_models import (
    CustomGymAction,
    CustomGymObservation,
    CustomGymState,
)


class CustomBrowserGymEnvironment:
    """Base class for custom BrowserGym environments.

    This class provides the basic Gym-like interface (reset, step, close)
    but uses Playwright directly instead of going through BrowserGym's
    registration system.

    To create a custom task:
    1. Subclass this class
    2. Implement _get_task_url() to return the starting URL
    3. Implement _extract_observation() to parse page state
    4. Implement _calculate_reward() to compute rewards
    5. Implement _check_done() to determine episode termination
    """

    def __init__(
        self,
        task_name: str,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout: float = 10000.0,
        max_steps: int = 50,
        **kwargs: Any,
    ):
        """Initialize the custom environment.

        Args:
            task_name: Name of your custom task
            headless: Whether to run browser in headless mode
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            timeout: Action timeout in milliseconds
            max_steps: Maximum steps per episode
            **kwargs: Additional custom parameters
        """
        self.task_name = task_name
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout = timeout
        self.max_steps = max_steps
        self.custom_params = kwargs

        # Playwright objects (initialized in reset)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # State tracking
        self._state = CustomGymState(
            episode_id=str(uuid4()),
            step_count=0,
            benchmark="custom",
            task_name=task_name,
            max_steps=max_steps,
        )

    @abstractmethod
    def _get_task_url(self) -> str:
        """Get the starting URL for this task.

        Returns:
            URL to navigate to when resetting the environment
        """
        pass

    @abstractmethod
    def _get_goal_description(self) -> str:
        """Get the goal/instruction for this task.

        Returns:
            Human-readable description of the task goal
        """
        pass

    @abstractmethod
    async def _extract_observation(self, page: Page) -> Dict[str, Any]:
        """Extract observation data from the current page state.

        Args:
            page: Playwright Page object

        Returns:
            Dictionary with observation data (text, axtree_txt, etc.)
        """
        pass

    @abstractmethod
    def _calculate_reward(
        self, page_data: Dict[str, Any], action: str, error: Optional[str] = None
    ) -> float:
        """Calculate reward for the current step.

        Args:
            page_data: Data extracted from _extract_observation
            action: Action that was executed
            error: Error message if action failed

        Returns:
            Reward value
        """
        pass

    @abstractmethod
    def _check_done(self, page_data: Dict[str, Any]) -> bool:
        """Check if the episode should terminate.

        Args:
            page_data: Data extracted from _extract_observation

        Returns:
            True if episode should end, False otherwise
        """
        pass

    def _get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for async operations."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    async def _async_reset(self, seed: Optional[int] = None) -> CustomGymObservation:
        """Async implementation of reset."""
        # Generate new episode ID
        self._state = CustomGymState(
            episode_id=str(uuid4()),
            step_count=0,
            benchmark="custom",
            task_name=self.task_name,
            max_steps=self.max_steps,
        )

        # Initialize Playwright if needed
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless
            )

        # Create new page
        if self._page:
            await self._page.close()

        self._page = await self._browser.new_page(
            viewport={
                "width": self.viewport_width,
                "height": self.viewport_height,
            }
        )

        # Set timeout
        self._page.set_default_timeout(self.timeout)

        # Navigate to task URL
        task_url = self._get_task_url()
        await self._page.goto(task_url)

        # Extract initial observation
        page_data = await self._extract_observation(self._page)
        goal = self._get_goal_description()

        self._state.current_url = self._page.url
        self._state.goal = goal

        return CustomGymObservation(
            text=page_data.get("text", ""),
            url=self._page.url,
            screenshot=page_data.get("screenshot"),
            goal=goal,
            axtree_txt=page_data.get("axtree_txt", ""),
            pruned_html=page_data.get("pruned_html", ""),
            error="",
            last_action_error=False,
            done=False,
            reward=0.0,
            custom_data=page_data.get("custom_data"),
        )

    async def _async_step(self, action_str: str) -> CustomGymObservation:
        """Async implementation of step."""
        self._state.step_count += 1

        error_msg = ""
        last_action_error = False

        try:
            # Execute the action
            # BrowserGym actions are Python-like function calls
            # We need to parse and execute them
            await self._execute_action(action_str)

        except Exception as e:
            error_msg = str(e)
            last_action_error = True

        # Extract observation
        page_data = await self._extract_observation(self._page)

        # Calculate reward
        reward = self._calculate_reward(page_data, action_str, error_msg)
        self._state.cum_reward += reward

        # Check if done
        done = self._check_done(page_data) or self._state.step_count >= self.max_steps

        # Update state
        self._state.current_url = self._page.url

        return CustomGymObservation(
            text=page_data.get("text", ""),
            url=self._page.url,
            screenshot=page_data.get("screenshot"),
            goal=self._state.goal,
            axtree_txt=page_data.get("axtree_txt", ""),
            pruned_html=page_data.get("pruned_html", ""),
            error=error_msg,
            last_action_error=last_action_error,
            done=done,
            reward=reward,
            custom_data=page_data.get("custom_data"),
        )

    async def _execute_action(self, action_str: str) -> None:
        """Execute a BrowserGym-style action string.

        Args:
            action_str: Action string like "click('button')" or "fill('input', 'text')"
        """
        # Simple action parser - you can make this more sophisticated
        action_str = action_str.strip()

        if action_str.startswith("click("):
            # Extract selector from click('selector')
            selector = action_str[6:-1].strip("'\"")
            await self._page.click(selector)

        elif action_str.startswith("fill("):
            # Extract selector and text from fill('selector', 'text')
            parts = action_str[5:-1].split(",", 1)
            selector = parts[0].strip().strip("'\"")
            text = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
            await self._page.fill(selector, text)

        elif action_str.startswith("goto("):
            # Extract URL from goto('url')
            url = action_str[5:-1].strip("'\"")
            await self._page.goto(url)

        elif action_str.startswith("press("):
            # Extract key from press('key')
            key = action_str[6:-1].strip("'\"")
            await self._page.keyboard.press(key)

        elif action_str.startswith("scroll("):
            # Extract direction from scroll('direction')
            direction = action_str[7:-1].strip("'\"")
            if direction == "down":
                await self._page.mouse.wheel(0, 500)
            elif direction == "up":
                await self._page.mouse.wheel(0, -500)

        else:
            # Try to execute as JavaScript if not recognized
            await self._page.evaluate(action_str)

    def reset(self, seed: Optional[int] = None) -> CustomGymObservation:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation
        """
        loop = self._get_or_create_event_loop()
        return loop.run_until_complete(self._async_reset(seed))

    def step(self, action: CustomGymAction) -> CustomGymObservation:
        """Execute an action.

        Args:
            action: Action to execute

        Returns:
            Observation after executing the action
        """
        loop = self._get_or_create_event_loop()
        return loop.run_until_complete(self._async_step(action.action_str))

    @property
    def state(self) -> CustomGymState:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up environment resources."""

        async def _async_close():
            if self._page:
                await self._page.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()

        if self._playwright:
            loop = self._get_or_create_event_loop()
            loop.run_until_complete(_async_close())
