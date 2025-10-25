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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .client_types import StepResult
from .containers.runtime import LocalDockerProvider

if TYPE_CHECKING:
    from .containers.runtime import ContainerProvider

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
        self._headers = default_headers or {}
        self._provider = provider

        # Configure session with retry logic and connection pooling
        self._http = self._create_session()

    def _create_session(self) -> requests.Session:
        """
        Create a requests Session with proper retry logic and connection pooling.

        This fixes timeout issues that occur after many sequential requests by:
        1. Adding automatic retry logic for connection errors
        2. Configuring connection pool with appropriate limits
        3. Handling stale connections gracefully
        4. Setting proper keep-alive behavior

        Returns:
            Configured requests.Session instance
        """
        session = requests.Session()

        # Configure retry strategy
        # Retry on connection errors and safe proxy/gateway errors
        # Note: We exclude 500 (Internal Server Error) to avoid retrying POST requests
        # that may have partially executed on the server, which could cause duplicate actions.
        # We only retry 502/503/504 which typically indicate proxy/gateway issues where
        # the request likely didn't reach the application server.
        retry_strategy = Retry(
            total=3,  # Maximum number of retries
            backoff_factor=0.3,  # Wait 0.3s, 0.6s, 1.2s between retries
            status_forcelist=[502, 503, 504],  # Retry on proxy/gateway errors (safer than 500)
            allowed_methods=["GET", "POST"],  # Retry these HTTP methods
            raise_on_status=False,  # Don't raise exception on failed retries (let caller handle)
        )

        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,  # Number of connection pools to cache (increase from default 10)
            pool_maxsize=20,  # Max connections per pool (increase from default 10)
            pool_block=False,  # Don't block when pool is full, create new connection
        )

        # Mount adapter for both http and https
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set keep-alive header to help with connection reuse
        session.headers.update({"Connection": "keep-alive"})

        return session

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

        try:
            # 2. Wait for server to be ready
            provider.wait_for_ready(base_url)

            # 3. Create and return client instance with provider reference
            return cls(base_url=base_url, provider=provider)

        except Exception:
            # If wait_for_ready fails or client creation fails, cleanup the container
            # to avoid leaving orphaned containers running
            try:
                provider.stop_container()
            except Exception:
                # If cleanup also fails, log but don't hide original error
                pass
            raise

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
        and remove the associated container. Also closes the HTTP session
        to release connection pool resources.
        """
        # Close HTTP session to release connections
        if hasattr(self, '_http') and self._http is not None:
            self._http.close()

        # Stop container if managed by provider
        if self._provider is not None:
            self._provider.stop_container()
