# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Core components for agentic environments."""

from __future__ import annotations

from importlib import import_module

from . import env_server
from .env_server import *  # noqa: F403

__all__ = [
    "EnvClient",
    "SyncEnvClient",
    "GenericEnvClient",
    "GenericAction",
    "MCPClientBase",
    "MCPToolClient",
] + env_server.__all__  # type: ignore


_LAZY_ATTRS = {
    "EnvClient": (".env_client", "EnvClient"),
    "SyncEnvClient": (".sync_client", "SyncEnvClient"),
    "GenericEnvClient": (".generic_client", "GenericEnvClient"),
    "GenericAction": (".generic_client", "GenericAction"),
    "MCPClientBase": (".mcp_client", "MCPClientBase"),
    "MCPToolClient": (".mcp_client", "MCPToolClient"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    try:
        value = getattr(env_server, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
