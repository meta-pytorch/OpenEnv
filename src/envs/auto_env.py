# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoEnv - Automatic Environment Selection
==========================================

AutoEnv provides a HuggingFace-style API for automatically selecting and
instantiating the correct environment client from installed packages or
HuggingFace Hub.

This module simplifies environment creation by automatically detecting the
environment type from the name and instantiating the appropriate client class.

Example:
    >>> from envs import AutoEnv, AutoAction
    >>>
    >>> # From installed package
    >>> env = AutoEnv.from_name("coding-env")
    >>>
    >>> # From HuggingFace Hub
    >>> env = AutoEnv.from_name("meta-pytorch/coding-env")
    >>>
    >>> # With configuration
    >>> env = AutoEnv.from_name("coding", env_vars={"DEBUG": "1"})
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING, Dict

from ._discovery import get_discovery, _is_hub_url, _normalize_env_name

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider
    from core.http_env_client import HTTPEnvClient

logger = logging.getLogger(__name__)


class AutoEnv:
    """
    AutoEnv automatically selects and instantiates the correct environment client
    based on environment names or HuggingFace Hub repositories.

    This class follows the HuggingFace AutoModel pattern, making it easy to work
    with different environments without needing to import specific client classes.

    The class provides factory methods that:
    1. Check if name is a HuggingFace Hub URL/repo ID
    2. If Hub: download and install the environment package
    3. If local: look up the installed openenv-* package
    4. Import and instantiate the client class

    Example:
        >>> # From installed package
        >>> env = AutoEnv.from_name("coding-env")
        >>>
        >>> # From HuggingFace Hub
        >>> env = AutoEnv.from_name("meta-pytorch/coding-env")
        >>>
        >>> # List available environments
        >>> AutoEnv.list_environments()

    Note:
        AutoEnv is not meant to be instantiated directly. Use the class method
        from_name() instead.
    """

    def __init__(self):
        """AutoEnv should not be instantiated directly. Use class methods instead."""
        raise TypeError(
            "AutoEnv is a factory class and should not be instantiated directly. "
            "Use AutoEnv.from_name() instead."
        )

    @classmethod
    def _download_from_hub(
        cls, repo_id: str, cache_dir: Optional[Path] = None
    ) -> Path:
        """
        Download environment from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "meta-pytorch/coding-env")
            cache_dir: Optional cache directory

        Returns:
            Path to downloaded environment directory

        Raises:
            ImportError: If huggingface_hub is not installed
            ValueError: If download fails
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "HuggingFace Hub support requires huggingface_hub package.\n"
                "Install it with: pip install huggingface_hub"
            )

        # Clean up repo_id if it's a full URL
        if "huggingface.co" in repo_id:
            # Extract org/repo from URL
            # https://huggingface.co/meta-pytorch/coding-env -> meta-pytorch/coding-env
            parts = repo_id.split("/")
            if len(parts) >= 2:
                repo_id = f"{parts[-2]}/{parts[-1]}"

        logger.info(f"Downloading environment from HuggingFace Hub: {repo_id}")

        try:
            # Download to cache
            env_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir or Path(tempfile.gettempdir()) / "openenv_hub_cache",
                repo_type="space",  # OpenEnv environments are published as Spaces
            )
            return Path(env_path)
        except Exception as e:
            raise ValueError(
                f"Failed to download environment from HuggingFace Hub: {repo_id}\n"
                f"Error: {e}\n"
                f"Make sure the repository exists and is accessible."
            ) from e

    @classmethod
    def _install_from_path(cls, env_path: Path) -> str:
        """
        Install environment package from a local path.

        Args:
            env_path: Path to environment directory containing pyproject.toml

        Returns:
            Package name that was installed

        Raises:
            ValueError: If installation fails
        """
        if not (env_path / "pyproject.toml").exists():
            raise ValueError(
                f"Environment directory does not contain pyproject.toml: {env_path}"
            )

        logger.info(f"Installing environment from: {env_path}")

        try:
            # Install in editable mode
            subprocess.run(
                ["pip", "install", "-e", str(env_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            # Read package name from pyproject.toml
            import toml

            with open(env_path / "pyproject.toml", "r") as f:
                pyproject = toml.load(f)

            package_name = pyproject.get("project", {}).get("name")
            if not package_name:
                raise ValueError("Could not determine package name from pyproject.toml")

            logger.info(f"Successfully installed: {package_name}")
            return package_name

        except subprocess.CalledProcessError as e:
            raise ValueError(
                f"Failed to install environment package from {env_path}\n"
                f"Error: {e.stderr}"
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to install environment package: {e}") from e

    @classmethod
    def from_name(
        cls,
        name: str,
        base_url: Optional[str] = None,
        docker_image: Optional[str] = None,
        container_provider: Optional[ContainerProvider] = None,
        wait_timeout: float = 30.0,
        env_vars: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HTTPEnvClient:
        """
        Create an environment client from a name or HuggingFace Hub repository.

        This method automatically:
        1. Checks if the name is a HuggingFace Hub URL/repo ID
        2. If Hub: downloads and installs the environment package
        3. If local: looks up the installed openenv-* package
        4. Imports the client class and instantiates it

        Args:
            name: Environment name or HuggingFace Hub repo ID
                  Examples:
                  - "coding" / "coding-env" / "coding_env"
                  - "meta-pytorch/coding-env" (Hub repo ID)
                  - "https://huggingface.co/meta-pytorch/coding-env" (Hub URL)
            base_url: Optional base URL for HTTP connection
            docker_image: Optional Docker image name (overrides default)
            container_provider: Optional container provider
            wait_timeout: Timeout for container startup (seconds)
            env_vars: Optional environment variables for the container
            **kwargs: Additional arguments passed to the client class

        Returns:
            Instance of the environment client class

        Raises:
            ValueError: If environment not found or cannot be loaded
            ImportError: If environment package is not installed

        Examples:
            >>> # From installed package
            >>> env = AutoEnv.from_name("coding-env")
            >>>
            >>> # From HuggingFace Hub
            >>> env = AutoEnv.from_name("meta-pytorch/coding-env")
            >>>
            >>> # With custom Docker image
            >>> env = AutoEnv.from_name("coding", docker_image="my-coding-env:v2")
            >>>
            >>> # With environment variables
            >>> env = AutoEnv.from_name(
            ...     "dipg",
            ...     env_vars={"DIPG_DATASET_PATH": "/data/dipg"}
            ... )
        """
        # Check if it's a HuggingFace Hub URL or repo ID
        if _is_hub_url(name):
            # Download from Hub and install
            env_path = cls._download_from_hub(name)
            package_name = cls._install_from_path(env_path)

            # Clear discovery cache to pick up the newly installed package
            get_discovery().clear_cache()

            # Extract environment name from package name
            # "openenv-coding_env" -> "coding_env"
            env_name = package_name.replace("openenv-", "").replace("-", "_")
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
                    f"Or specify a HuggingFace Hub repository: AutoEnv.from_name('org/repo')"
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

        # Get the client class
        try:
            client_class = env_info.get_client_class()
        except ImportError as e:
            raise ImportError(
                f"Failed to import environment client for '{env_name}'.\n"
                f"Package '{env_info.package_name}' appears to be installed but the module cannot be imported.\n"
                f"Try reinstalling: pip install --force-reinstall {env_info.package_name}\n"
                f"Original error: {e}"
            ) from e

        # Determine Docker image to use
        if docker_image is None:
            docker_image = env_info.default_image

        # Create client instance
        try:
            if base_url:
                # Connect to existing server at URL
                return client_class(base_url=base_url, **kwargs)
            else:
                # Start new Docker container
                return client_class.from_docker_image(
                    image=docker_image,
                    container_provider=container_provider,
                    wait_timeout=wait_timeout,
                    env_vars=env_vars or {},
                    **kwargs,
                )
        except Exception as e:
            raise ValueError(
                f"Failed to create environment client for '{env_name}'.\n"
                f"Client class: {client_class.__name__}\n"
                f"Docker image: {docker_image}\n"
                f"Error: {e}"
            ) from e

    @classmethod
    def get_env_class(cls, name: str):
        """
        Get the environment client class without instantiating it.

        Args:
            name: Environment name

        Returns:
            The environment client class

        Raises:
            ValueError: If environment not found

        Examples:
            >>> CodingEnv = AutoEnv.get_env_class("coding")
            >>> # Now you can instantiate it yourself
            >>> env = CodingEnv(base_url="http://localhost:8000")
        """
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(name)

        if not env_info:
            raise ValueError(f"Unknown environment: {name}")

        return env_info.get_client_class()

    @classmethod
    def get_env_info(cls, name: str) -> Dict[str, Any]:
        """
        Get detailed information about an environment.

        Args:
            name: Environment name

        Returns:
            Dictionary with environment metadata

        Raises:
            ValueError: If environment not found

        Examples:
            >>> info = AutoEnv.get_env_info("coding")
            >>> print(info['description'])
            'Coding environment for OpenEnv'
            >>> print(info['default_image'])
            'coding-env:latest'
        """
        discovery = get_discovery()
        env_info = discovery.get_environment_by_name(name)

        if not env_info:
            raise ValueError(f"Unknown environment: {name}")

        return {
            "env_key": env_info.env_key,
            "name": env_info.name,
            "package": env_info.package_name,
            "version": env_info.version,
            "description": env_info.description,
            "env_class": env_info.client_class_name,
            "action_class": env_info.action_class_name,
            "observation_class": env_info.observation_class_name,
            "module": env_info.client_module_path,
            "default_image": env_info.default_image,
            "spec_version": env_info.spec_version,
        }

    @classmethod
    def list_environments(cls) -> None:
        """
        Print a formatted list of all available environments.

        This discovers all installed openenv-* packages and displays
        their metadata in a user-friendly format.

        Examples:
            >>> AutoEnv.list_environments()
            Available OpenEnv Environments:
            ----------------------------------------------------------------------
              echo           : Echo Environment (v0.1.0)
                               Package: openenv-echo-env
              coding         : Coding Environment (v0.1.0)
                               Package: openenv-coding_env
            ----------------------------------------------------------------------
            Total: 2 environments
        """
        discovery = get_discovery()
        discovery.list_environments()
