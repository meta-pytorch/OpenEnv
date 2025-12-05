# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoAction - Automatic Action Class Selection
==============================================

AutoAction provides a HuggingFace-style API for automatically retrieving the
correct Action class from installed packages or HuggingFace Hub.

This module simplifies working with environment actions by automatically
detecting and returning the appropriate Action class without requiring
manual imports.

Example:
    >>> from envs import AutoEnv, AutoAction
    >>>
    >>> # Get Action class from environment name
    >>> CodeAction = AutoAction.from_name("coding")
    >>> action = CodeAction(code="print('Hello!')")
    >>>
    >>> # From HuggingFace Hub
    >>> CodeAction = AutoAction.from_name("meta-pytorch/coding-env")
    >>>
    >>> # Use with AutoEnv
    >>> env = AutoEnv.from_name("coding-env")
    >>> result = env.step(action)
"""

from __future__ import annotations

import logging
from typing import Type, Dict, Any

from ._discovery import get_discovery, _is_hub_url
from .auto_env import AutoEnv

logger = logging.getLogger(__name__)


class AutoAction:
    """
    AutoAction automatically retrieves the correct Action class based on
    environment names or HuggingFace Hub repositories.

    This class follows the HuggingFace AutoModel pattern, making it easy to
    get the right Action class without needing to know which module to import.

    The class provides factory methods that look up the Action class and
    return the class (not an instance) for you to instantiate.

    Example:
        >>> # From installed package
        >>> CodeAction = AutoAction.from_name("coding")
        >>> action = CodeAction(code="print('test')")
        >>>
        >>> # From HuggingFace Hub
        >>> CodeAction = AutoAction.from_name("meta-pytorch/coding-env")
        >>> action = CodeAction(code="print('test')")
        >>>
        >>> # Use with AutoEnv for a complete workflow
        >>> env = AutoEnv.from_name("coding-env")
        >>> ActionClass = AutoAction.from_name("coding-env")
        >>> action = ActionClass(code="print('Hello, AutoAction!')")
        >>> result = env.step(action)

    Note:
        AutoAction is not meant to be instantiated directly. Use the class
        method from_name() instead.
    """

    def __init__(self):
        """AutoAction should not be instantiated directly. Use class methods instead."""
        raise TypeError(
            "AutoAction is a factory class and should not be instantiated directly. "
            "Use AutoAction.from_name() instead."
        )

    @classmethod
    def from_name(cls, name: str) -> Type:
        """
        Get the Action class from environment name or HuggingFace Hub repository.

        This method automatically:
        1. Checks if the name is a HuggingFace Hub URL/repo ID
        2. If Hub: downloads and installs the environment package
        3. If local: looks up the installed openenv-* package
        4. Imports and returns the Action class

        Args:
            name: Environment name or HuggingFace Hub repo ID
                  Examples:
                  - "coding" / "coding-env" / "coding_env"
                  - "meta-pytorch/coding-env" (Hub repo ID)
                  - "https://huggingface.co/meta-pytorch/coding-env" (Hub URL)

        Returns:
            Action class (not an instance!)

        Raises:
            ValueError: If environment not found
            ImportError: If environment package is not installed

        Examples:
            >>> # From installed package
            >>> CodeAction = AutoAction.from_name("coding-env")
            >>> action = CodeAction(code="print('Hello!')")
            >>>
            >>> # From HuggingFace Hub
            >>> CodeAction = AutoAction.from_name("meta-pytorch/coding-env")
            >>> action = CodeAction(code="print('Hello!')")
            >>>
            >>> # Different name formats
            >>> EchoAction = AutoAction.from_name("echo")
            >>> EchoAction = AutoAction.from_name("echo-env")
            >>> EchoAction = AutoAction.from_name("echo_env")
        """
        # Check if it's a HuggingFace Hub URL or repo ID
        if _is_hub_url(name):
            # Ensure package is installed (reuse AutoEnv logic, downloads only if needed)
            env_name = AutoEnv._ensure_package_from_hub(name)
        else:
            env_name = name

        # Get environment info from discovery
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(env_name)

        if not env_info:
            # Environment not found - provide helpful error message
            available_envs = discovery.discover()

            if not available_envs:
                raise ValueError(
                    f"No OpenEnv environments found.\n"
                    f"Install an environment with: pip install openenv-<env-name>\n"
                    f"Or specify a HuggingFace Hub repository: AutoAction.from_name('org/repo')"
                )

            # Try to suggest similar environment names
            from difflib import get_close_matches

            env_keys = list(available_envs.keys())
            suggestions = get_close_matches(env_name, env_keys, n=3, cutoff=0.6)

            error_msg = f"Unknown environment '{env_name}'.\n"
            if suggestions:
                error_msg += f"Did you mean: {', '.join(suggestions)}?\n"
            error_msg += f"Available environments: {', '.join(sorted(env_keys))}"

            raise ValueError(error_msg)

        # Get the action class
        try:
            action_class = env_info.get_action_class()
            return action_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import action class for '{env_name}'.\n"
                f"Package '{env_info.package_name}' appears to be installed but the module cannot be imported.\n"
                f"Try reinstalling: pip install --force-reinstall {env_info.package_name}\n"
                f"Original error: {e}"
            ) from e

    @classmethod
    def from_env(cls, env_name: str) -> Type:
        """
        Get the Action class from environment name.

        This is an alias for from_name() for backward compatibility and clarity.

        Args:
            env_name: Environment name (e.g., "coding", "echo")

        Returns:
            Action class (not an instance!)

        Examples:
            >>> CodeAction = AutoAction.from_env("coding")
            >>> action = CodeAction(code="print('Hello!')")
        """
        return cls.from_name(env_name)

    @classmethod
    def get_action_info(cls, name: str) -> Dict[str, Any]:
        """
        Get detailed information about an action class.

        Args:
            name: Environment name

        Returns:
            Dictionary with action class metadata

        Raises:
            ValueError: If environment not found

        Examples:
            >>> info = AutoAction.get_action_info("coding")
            >>> print(info['action_class'])
            'CodingAction'
            >>> print(info['module'])
            'coding_env.client'
        """
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(name)

        if not env_info:
            raise ValueError(f"Unknown environment: {name}")

        return {
            "env_key": env_info.env_key,
            "env_name": env_info.name,
            "package": env_info.package_name,
            "action_class": env_info.action_class_name,
            "observation_class": env_info.observation_class_name,
            "module": env_info.client_module_path,
        }

    @classmethod
    def list_actions(cls) -> None:
        """
        Print a formatted list of all available action classes.

        This discovers all installed openenv-* packages and displays
        their action class information in a user-friendly format.

        Examples:
            >>> AutoAction.list_actions()
            Available Action Classes:
            ----------------------------------------------------------------------
              echo           : EchoAction (from openenv-echo-env)
              coding         : CodingAction (from openenv-coding_env)
            ----------------------------------------------------------------------
            Total: 2 action classes
        """
        discovery = get_discovery()
        environments = discovery.discover()

        print("Available Action Classes:")
        print("-" * 70)

        if not environments:
            print("  No OpenEnv environments found.")
            print("  Install environments with: pip install openenv-<env-name>")
        else:
            for env_key in sorted(environments.keys()):
                env = environments[env_key]
                print(f"  {env_key:<15}: {env.action_class_name}")
                print(f"                   Package: {env.package_name}")

        print("-" * 70)
        print(f"Total: {len(environments)} action classes")
