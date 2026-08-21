# SPDX-License-Identifier: BSD-3-Clause

"""Bundled openenvd sidecars.

Each sidecar is a plain ``TaskSpec`` factory plus a runnable module, so it
stacks on the daemon through the same registration API as any external
task — there is no privileged in-process path.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "trajectory_writer_spec":
        from openenv.core.openenvd.sidecars.trajectory_writer import (
            trajectory_writer_spec,
        )

        return trajectory_writer_spec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["trajectory_writer_spec"]
