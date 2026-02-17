# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grid World Environment - A simple test environment for HTTP server."""

__all__ = ["GridWorldAction", "GridWorldObservation", "GridWorldEnv"]


def __getattr__(name: str):
    if name == "GridWorldEnv":
        from .client import GridWorldEnv

        return GridWorldEnv
    if name == "GridWorldAction":
        from .models import GridWorldAction

        return GridWorldAction
    if name == "GridWorldObservation":
        from .models import GridWorldObservation

        return GridWorldObservation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
