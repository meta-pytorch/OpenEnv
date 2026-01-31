# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Protocol, TypedDict, TypeVar

from .types import Action, Observation, State, EnvironmentMetadata

ActT = TypeVar("ActT", bound=Action)
ObsT = TypeVar("ObsT", bound=Observation)
StateT = TypeVar("StateT", bound=State)


class Message(TypedDict):
    """A message in a conversation.

    Compatible with Huggingface chat template format.
    """

    role: str
    content: str


class ModelTokenizer(Protocol):
    """Protocol for tokenizers that support chat templates.

    This protocol defines the interface that tokenizers must implement
    to work with chat-based environments. It's compatible with
    Huggingface transformers tokenizers.
    """

    def apply_chat_template(
        self,
        conversation: list[Message],
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply a chat template to format and optionally tokenize a conversation.

        Args:
            conversation: List of message dictionaries with 'role' and 'content'
            tokenize: Whether to tokenize the output
            return_tensors: Format for returned tensors ('pt' for PyTorch)
            **kwargs: Additional arguments

        Returns:
            Formatted and optionally tokenized conversation
        """
        ...

    def decode(
        self, token_ids: Any, skip_special_tokens: bool = False, **kwargs: Any
    ) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            **kwargs: Additional arguments

        Returns:
            Decoded text string
        """
        ...


class Transform(ABC, Generic[ObsT]):
    """Transform observations to add rewards, metrics, or other modifications.

    Transforms follow the TorchRL pattern where they take an observation
    and return a (potentially modified) observation. This allows for
    flexible reward computation and observation augmentation.
    """

    @abstractmethod
    def __call__(self, observation: ObsT) -> ObsT:
        """Transform an observation.

        Args:
            observation: The input observation

        Returns:
            The transformed observation
        """
        pass


class Environment(ABC, Generic[ActT, ObsT, StateT]):
    """Base class for all environment servers following Gym/Gymnasium API.

    Args:
        transform: Optional transform to apply to observations

    Class Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: Whether this environment supports concurrent sessions.
            When True, multiple WebSocket connections can each have their own
            environment instance (up to max_concurrent_envs). When False (default),
            the environment should only be used with a single session at a time.

            Set this to True in your Environment subclass if:
            - The environment uses proper session isolation (e.g., unique working dirs)
            - No shared mutable state exists between instances
            - External resources (databases, APIs) can handle concurrent access
    """

    # Class-level flag indicating whether this environment supports concurrent sessions
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self, transform: Optional[Transform[ObsT]] = None):
        self.transform = transform

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ObsT:
        """Reset the environment and return initial observation."""
        pass

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ObsT:
        """Async version of reset. Default implementation calls sync reset.

        Override to provide true async implementation.
        """
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)

    @abstractmethod
    def step(
        self,
        action: ActT,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ObsT:
        """Take a step in the environment."""
        pass

    async def step_async(
        self,
        action: ActT,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ObsT:
        """Async version of step. Default implementation calls sync step.

        Override to provide true async implementation.
        """
        return self.step(action, timeout_s=timeout_s, **kwargs)

    @property
    @abstractmethod
    def state(self) -> StateT:
        """Get the current environment state."""
        pass

    def get_metadata(self) -> EnvironmentMetadata:
        """
        Get metadata about this environment.

        Override this method to provide custom metadata for the environment.
        Default implementation returns basic metadata derived from class name.

        Returns:
            EnvironmentMetadata with environment information
        """
        return EnvironmentMetadata(
            name=self.__class__.__name__,
            description=f"{self.__class__.__name__} environment",
            version="1.0.0",
        )

    def _apply_transform(self, observation: ObsT) -> ObsT:
        """Apply transform if one is provided."""
        if self.transform is not None:
            return self.transform(observation)
        return observation

    def close(self) -> None:
        """Clean up resources used by the environment.

        Override this method to implement custom cleanup logic.
        Called when the environment is being destroyed or reset.
        """
        pass
