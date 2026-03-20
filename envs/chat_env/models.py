# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Chat Environment.

The Chat environment provides a chat-based interface for LLMs with support
for tokenization and message history management.
"""

import torch
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


class ChatAction(Action):
    """Action for chat environments.

    Contains tokens that represent the action to be taken.
    This interfaces directly with models.
    """

    tokens: list[int] = Field(..., min_length=1)

    @field_validator("tokens", mode="before")
    @classmethod
    def _coerce_tokens(cls, value):
        """Accept either tensors or JSON arrays on the public HTTP surface."""
        if isinstance(value, torch.Tensor):
            value = value.flatten().tolist()
        elif hasattr(value, "tolist") and callable(value.tolist):
            value = value.tolist()

        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            return [int(token) for token in value]
        raise TypeError("tokens must be provided as a list of token ids")


class ChatState(State):
    """State of the ChatEnvironment containing message history."""

    # TODO: revert to list[Message] once openenv-core ships typing_extensions.TypedDict
    # in interfaces.py and chat_env/pyproject.toml pins to that release.
    history_messages: list[dict[str, str]] = Field(default_factory=list)
    history_tokens: list[torch.Tensor] = Field(
        default_factory=list
    )  # Same len as messages


class ChatObservation(Observation):
    """Observation returned by ChatEnvironment.

    Contains the message history in Huggingface format (list of dicts with role/content)
    and the tokenized representation of the entire conversation.

    The environment owns the tokenizer and generates the tokens from the messages.

    Example:
    messages = [
     {"role": "system", "content": "You are a helpful assistant"},
     {"role": "user", "content": "How tall is the Eiffel Tower?"},
    ]
    tokens = tensor([1, 2, 3, 4, 5, ...])  # tokenized entire conversation
    """

    # TODO: revert to list[Message] (same as above)
    messages: list[dict[str, str]] = Field(default_factory=list)
    tokens: list[int] = Field(default_factory=list)
    # Inherited Fields from Observation ABC: reward, done, metadata
