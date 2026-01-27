"""Registry for custom BrowserGym tasks.

This module provides a central place to register and retrieve custom tasks.
Add your custom tasks here to make them available through the BrowserGym environment.
"""

import os
from typing import Any, Dict

from .custom_base import CustomBrowserGymEnvironment


# Registry of custom tasks
_CUSTOM_TASKS: Dict[str, type] = {}


def register_custom_task(name: str, task_class: type) -> None:
    """Register a custom task.

    Args:
        name: Task name (e.g., 'copy-paste', 'data-entry')
        task_class: Class that extends CustomBrowserGymEnvironment
    """
    if not issubclass(task_class, CustomBrowserGymEnvironment):
        raise ValueError(
            f"Task class must extend CustomBrowserGymEnvironment, got {task_class}"
        )
    _CUSTOM_TASKS[name] = task_class


def get_custom_task(task_name: str, **kwargs: Any) -> CustomBrowserGymEnvironment:
    """Get a custom task instance.

    Args:
        task_name: Name of the task to retrieve
        **kwargs: Arguments to pass to the task constructor

    Returns:
        Instance of the custom task

    Raises:
        ValueError: If task is not registered
    """
    if task_name not in _CUSTOM_TASKS:
        available = ", ".join(_CUSTOM_TASKS.keys()) or "none"
        raise ValueError(
            f"Custom task '{task_name}' not found. "
            f"Available tasks: {available}. "
            f"Register your task using register_custom_task()."
        )

    task_class = _CUSTOM_TASKS[task_name]
    return task_class(task_name=task_name, **kwargs)


def list_custom_tasks() -> list[str]:
    """List all registered custom tasks.

    Returns:
        List of task names
    """
    return list(_CUSTOM_TASKS.keys())


# ============================================================================
# Copy-Paste in a single page HTML task
# ============================================================================


class CopyPasteTask(CustomBrowserGymEnvironment):
    """Copy text from one field and paste into another."""

    def _get_task_url(self) -> str:
        """Get the URL for the copy-paste task."""
        task_html = os.path.join(os.path.dirname(__file__), "tasks", "copy-paste.html")
        return f"file://{task_html}"

    def _get_goal_description(self) -> str:
        """Get the goal description."""
        return "Copy the text from the source field and paste it into the target field, then click Submit."

    async def _extract_observation(self, page) -> dict:
        """Extract observation from the page."""
        # Get the accessibility tree or HTML
        try:
            # Try to get the page content
            content = await page.content()

            # Get the current values of source and target fields
            source_value = await page.evaluate(
                "document.querySelector('#source-text')?.value || ''"
            )
            target_value = await page.evaluate(
                "document.querySelector('#target-text')?.value || ''"
            )

            # Get success message if visible
            success_msg = await page.evaluate(
                "document.querySelector('#success-message')?.textContent || ''"
            )

            return {
                "text": content,
                "pruned_html": content[:1000],  # Truncate for observation
                "custom_data": {
                    "source_value": source_value,
                    "target_value": target_value,
                    "success_message": success_msg,
                },
            }
        except Exception as e:
            return {
                "text": f"Error extracting observation: {e}",
                "custom_data": {"error": str(e)},
            }

    def _calculate_reward(
        self, page_data: dict, action: str, error: str | None = None
    ) -> float:
        """Calculate reward based on page state."""
        if error:
            return -0.1  # Small penalty for errors

        custom_data = page_data.get("custom_data", {})

        # Check if task is completed successfully
        if "Success!" in custom_data.get("success_message", ""):
            return 1.0

        # Partial reward if text is copied correctly
        source = custom_data.get("source_value", "")
        target = custom_data.get("target_value", "")

        if source and target and source == target:
            return 0.5

        return 0.0

    def _check_done(self, page_data: dict) -> bool:
        """Check if the task is complete."""
        custom_data = page_data.get("custom_data", {})
        # Task is done if success message is shown
        return "Success!" in custom_data.get("success_message", "")


# Register the example task
register_custom_task("copy-paste", CopyPasteTask)


# ============================================================================
# Multi-Tab Copy-Paste Task
# ============================================================================


class CopyPasteMultiTabTask(CustomBrowserGymEnvironment):
    """Copy text from one tab and paste it into another tab.

    This task demonstrates handling multiple browser tabs/pages.
    The agent needs to:
    1. Copy text from the source page (tab 1)
    2. Navigate/switch to the target page (tab 2)
    3. Paste the text into the target field
    4. Submit the form
    """

    def _get_task_url(self) -> str:
        """Get the URL for the first tab (source page)."""

        task_html = os.path.join(
            os.path.dirname(__file__), "tasks", "copy-paste-source.html"
        )
        return f"file://{task_html}"

    def _get_goal_description(self) -> str:
        """Get the goal description."""
        return (
            "Copy the text from the source page, then navigate to the target page "
            "(click 'Open Target Page' button), paste the text into the input field, "
            "and click Submit."
        )

    async def _extract_observation(self, page) -> dict:
        """Extract observation from the current page."""
        try:
            content = await page.content()
            current_url = page.url

            # Determine which page we're on
            if "source" in current_url:
                # On source page
                source_value = await page.evaluate(
                    "document.querySelector('#source-text')?.textContent || ''"
                )

                return {
                    "text": content,
                    "pruned_html": content[:1000],
                    "custom_data": {
                        "current_page": "source",
                        "source_value": source_value,
                        "task_step": "copy_from_source",
                    },
                }

            elif "target" in current_url:
                # On target page
                target_value = await page.evaluate(
                    "document.querySelector('#target-text')?.value || ''"
                )
                success_msg = await page.evaluate(
                    "document.querySelector('#success-message')?.textContent || ''"
                )

                return {
                    "text": content,
                    "pruned_html": content[:1000],
                    "custom_data": {
                        "current_page": "target",
                        "target_value": target_value,
                        "success_message": success_msg,
                        "task_step": "paste_to_target",
                    },
                }

            else:
                # Unknown page
                return {
                    "text": content,
                    "custom_data": {
                        "current_page": "unknown",
                        "error": "Not on source or target page",
                    },
                }

        except Exception as e:
            return {
                "text": f"Error extracting observation: {e}",
                "custom_data": {"error": str(e)},
            }

    def _calculate_reward(
        self, page_data: dict, action: str, error: str | None = None
    ) -> float:
        """Calculate reward based on page state and action."""
        if error:
            return -0.1

        custom_data = page_data.get("custom_data", {})
        current_page = custom_data.get("current_page", "")

        # Big reward for completing the task
        if "Success!" in custom_data.get("success_message", ""):
            return 1.0

        # Small reward for successfully navigating to target page
        if current_page == "target" and "goto" in action.lower():
            return 0.3

        # Medium reward if text is pasted correctly in target
        if current_page == "target":
            target_value = custom_data.get("target_value", "")
            # The expected text from source page
            if target_value and "Hello from the source page!" in target_value:
                return 0.6

        return 0.0

    def _check_done(self, page_data: dict) -> bool:
        """Check if the task is complete."""
        custom_data = page_data.get("custom_data", {})
        return "Success!" in custom_data.get("success_message", "")


# Register the multi-tab task
register_custom_task("copy-paste-multitab", CopyPasteMultiTabTask)


# ============================================================================
# Add your own custom tasks below by:
# 1. Creating a class that extends CustomBrowserGymEnvironment
# 2. Implementing the required methods
# 3. Registering it with register_custom_task()
# ============================================================================

# Example:
# class MyCustomTask(CustomBrowserGymEnvironment):
#     def _get_task_url(self) -> str:
#         return "https://my-task-url.com"
#
#     def _get_goal_description(self) -> str:
#         return "Do something amazing"
#
#     async def _extract_observation(self, page) -> dict:
#         return {"text": await page.content()}
#
#     def _calculate_reward(self, page_data, action, error=None) -> float:
#         return 1.0 if some_condition else 0.0
#
#     def _check_done(self, page_data) -> bool:
#         return some_completion_check
#
# register_custom_task("my-task", MyCustomTask)
