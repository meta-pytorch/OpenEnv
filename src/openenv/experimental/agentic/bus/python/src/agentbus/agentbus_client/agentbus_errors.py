# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Exceptions for AgentBus client operations.
"""


class AgentBusClientError(Exception):
    """Raised when an error occurs in the AgentBus client."""

    pass


class CodeSafetyError(Exception):
    """Raised when code is deemed unsafe for execution."""

    pass
