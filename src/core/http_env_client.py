"""
core/runner_env.py
Minimal HTTP-based environment client.
- Talks to a single env worker exposing: POST /reset, POST /step

Future hooks (commented below) for:
- episode_id, seed on reset
- request_id on step
- custom headers (auth/trace)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TYPE_CHECKING, TypeVar

import requests

from .client_types import StepResult
from .containers.runtime import LocalDockerProvider, UVProvider

if TYPE_CHECKING:
    from .containers.runtime import ContainerProvider, RuntimeProvider

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
EnvClientT = TypeVar("EnvClientT", bound="HTTPEnvClient")


class HTTPEnvClient(ABC, Generic[ActT, ObsT]):
    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 15.0,
        default_headers: Optional[Dict[str, str]] = None,
        provider: Optional["ContainerProvider"] = None,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = float(request_timeout_s)
        self._http = requests.Session()
        self._headers = default_headers or {}
        self._provider = provider

    @property
    def base_url(self) -> str:
        """Base URL of the connected environment server."""
        return self._base

    @classmethod
    def from_docker_image(
        cls: Type[EnvClientT],
        image: str,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> EnvClientT:
        """
        Create an environment client by spinning up a Docker container locally.

        This is a development utility that:
        1. Starts a Docker container from the specified image
        2. Waits for the server to be ready
        3. Creates and returns a client instance connected to the container

        Note: The container lifecycle management is left to the user or higher-level
        orchestration. The container will keep running until manually stopped.

        Args:
            image: Docker image name to run (e.g., "echo-env:latest")
            provider: Container provider to use (defaults to LocalDockerProvider)
            **kwargs: Additional arguments to pass to provider.start_container()
                     (e.g., env_vars, port)

        Returns:
            An instance of the client class connected to the running container

        Example:
            >>> from envs.coding_env.client import CodingEnv
            >>> from envs.coding_env.models import CodeAction
            >>>
            >>> # Create environment from image
            >>> env = CodingEnv.from_docker_image("coding-env:latest")
            >>>
            >>> # Create environment with custom env vars
            >>> env = CodingEnv.from_docker_image(
            ...     "coding-env:latest",
            ...     env_vars={"MY_VAR": "value"}
            ... )
            >>>
            >>> # Use the environment
            >>> result = env.reset()
            >>> print(result.observation)
            >>>
            >>> step_result = env.step(CodeAction(code="print('hello')"))
            >>> print(step_result.observation.stdout)
            >>>
            >>> # Cleanup (optional)
            >>> env.close()
        """

        # Use default provider if none provided
        if provider is None:
            provider = LocalDockerProvider()

        # 1. Start container with optional kwargs (e.g., env_vars, port)
        base_url = provider.start_container(image, **kwargs)

        # 2. Wait for server to be ready
        provider.wait_for_ready(base_url)

        # 3. Create and return client instance with provider reference
        return cls(base_url=base_url, provider=provider)

    @classmethod
    def from_hub(
        cls: Type[EnvClientT],
        repo_id: str,
        *,
        use_docker: bool = True,
        provider: Optional["ContainerProvider" | "RuntimeProvider"] = None,
        **provider_kwargs: Any,
    ) -> EnvClientT:
        """Create a client from a Hugging Face Space.

        Args:
            repo_id: Hugging Face space identifier ``{org}/{space}``.
            use_docker: When ``True`` (default) pull from the HF registry and
                launch via :class:`LocalDockerProvider`. When ``False`` run the
                space locally with :class:`UVProvider`.
            provider: Optional provider instance to reuse. Must be a
                :class:`ContainerProvider` when ``use_docker=True`` and a
                :class:`RuntimeProvider`` otherwise.
            provider_kwargs: Additional keyword arguments forwarded to either the
                container provider's ``start_container`` (docker) or to the
                ``UVProvider`` constructor/start (uv). When ``use_docker=False``,
                the ``project_path`` argument can be used to override the default
                git URL (``git+https://huggingface.co/spaces/{repo_id}``).
        """

        start_args = {}
        for key in ("port", "env_vars", "workers"):
            if key in provider_kwargs:
                start_args[key] = provider_kwargs.pop(key)

        if use_docker:
            docker_provider = provider or LocalDockerProvider()
            tag = provider_kwargs.pop("tag", "latest")
            image = f"registry.hf.space/{repo_id.replace('/', '-')}:{tag}"
            base_url = docker_provider.start_container(image, **start_args, **provider_kwargs)
            docker_provider.wait_for_ready(base_url)
            return cls(base_url=base_url, provider=docker_provider)
        else:
            if provider is None:
                uv_kwargs = dict(provider_kwargs)
                project_path = uv_kwargs.pop("project_path", None)
                if project_path is None:
                    project_path = f"git+https://huggingface.co/spaces/{repo_id}"

                provider = UVProvider(project_path=project_path, **uv_kwargs)
            else:
                if provider_kwargs:
                    raise ValueError("provider_kwargs cannot be used when supplying a provider instance")

            base_url = provider.start(**start_args)
            provider.wait_for_ready()

            return cls(base_url=base_url, provider=provider)

    @abstractmethod
    def _step_payload(self, action: ActT) -> dict:
        """Convert an Action object to the JSON body expected by the env server."""
        raise NotImplementedError

    @abstractmethod
    def _parse_result(self, payload: dict) -> StepResult[ObsT]:
        """Convert a JSON response from the env server to StepResult[ObsT]."""
        raise NotImplementedError

    @abstractmethod
    def _parse_state(self, payload: dict) -> Any:
        """Convert a JSON response from the state endpoint to a State object."""
        raise NotImplementedError

    # ---------- Environment Server Interface Methods ----------
    def reset(self) -> StepResult[ObsT]:
        body: Dict[str, Any] = {}
        # TODO: later:
        # body["seed"] = seed
        # body["episode_id"] = episode_id
        r = self._http.post(
            f"{self._base}/reset",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_result(r.json())

    def step(self, action: ActT) -> StepResult[ObsT]:
        body: Dict[str, Any] = {
            "action": self._step_payload(action),
            "timeout_s": int(self._timeout),
        }
        # TODO: later:
        # body["request_id"] = str(uuid.uuid4())
        # body["episode_id"] = current_episode_id
        r = self._http.post(
            f"{self._base}/step",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_result(r.json())

    def state(self) -> Any:
        """
        Get the current environment state from the server.

        Returns:
            State object with environment state information (e.g., episode_id, step_count)

        Example:
            >>> client = EchoEnv.from_docker_image("echo-env:latest")
            >>> result = client.reset()
            >>> state = client.state()
            >>> print(state.episode_id)
            >>> print(state.step_count)
        """
        r = self._http.get(
            f"{self._base}/state",
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_state(r.json())

    def close(self) -> None:
        """
        Close the environment and clean up resources.

        If this client was created via from_docker_image(), this will stop
        and remove the associated container.
        """
        if self._provider is not None:
            self._provider.stop_container()
