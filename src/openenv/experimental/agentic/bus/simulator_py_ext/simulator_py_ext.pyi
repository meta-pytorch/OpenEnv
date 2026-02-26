# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""Type stubs for simulator_py_ext Rust extension."""

from typing import Any

class SimulatorPyExt:
    """Python extension wrapper for the Rust Simulator.

    This is a thin bridge that allows Python's RustBackedEventLoop to schedule
    callbacks on the Rust simulator. The callbacks are stored in Python (by handle_id),
    and Rust calls `_fire_handle(handle_id)` when the scheduled time arrives.
    """

    def __new__(cls, seed: int) -> "SimulatorPyExt":
        """Create a new simulator with the given seed."""
        ...

    def schedule_callback(
        self,
        delay_ns: int,
        handle_id: int,
        event_loop: Any,
    ) -> None:
        """Schedule a callback to fire after delay_ns nanoseconds.

        When the delay expires, Rust will call `event_loop._fire_handle(handle_id)`.
        """
        ...

    def current_time_ns(self) -> int:
        """Get the current virtual time in nanoseconds since simulator start."""
        ...

    def run(self) -> None:
        """Run the simulator until all tasks complete."""
        ...
