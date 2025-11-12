# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the WebSearch Env Environment.

The WebSearch_env environment is a simple test environment that echoes back messages.
"""

from __future__ import annotations

# Use pydantic dataclass for validation
from pydantic.dataclasses import dataclass
from pydantic import Field
from core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class WebSearchAction(Action):
    """Action for the WebSearch Env environment - just a message to echo."""

    query: str = Field(..., description="The query to search the web for")


@dataclass(kw_only=True)
class WebSearchObservation(Observation):
    """Observation from the WebSearch Env environment - the echoed message."""

    content: str = Field(..., description="The formatted content of the search results or error message if the search failed")
    web_contents: list[WebContent] = Field(..., description="The web contents of the search results")


@dataclass(kw_only=True)
class WebContent:
    """Web content of a search result."""

    title: str = Field(..., description="The title of the web content")
    content: str = Field(..., description="The content of the web content")
    url: str = Field(..., description="The URL of the web content")
