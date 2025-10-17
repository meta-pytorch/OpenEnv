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

from dataclasses import dataclass, field

import torch

from core.env_server.interfaces import Message
from core.env_server.types import Action, Observation, State


@dataclass
class ChatAction(Action):
    """Action for chat environments.

    Contains tokens that represent the action to be taken.
    This interfaces directly with models.
    """

    tokens: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

    def __post_init__(self):
        """Validate required fields after initialization."""
        if self.tokens.numel() == 0:
            raise ValueError("tokens is required and cannot be empty")


@dataclass
class ChatState(State):
    """State of the ChatEnvironment containing message history."""

    history_messages: list[Message] = field(default_factory=list)
    history_tokens: list[torch.Tensor] = field(
        default_factory=list
    )  # Same len as messages


@dataclass(kw_only=True)
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

    messages: list[Message] = field(default_factory=list)
    tokens: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    # Inherited fields from Observation ABC: reward, done, metadata
