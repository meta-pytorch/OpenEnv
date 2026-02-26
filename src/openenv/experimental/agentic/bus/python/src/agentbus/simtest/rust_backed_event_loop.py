# pyre-strict
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Rust-backed asyncio event loop for deterministic testing.

This module provides RustBackedEventLoop, an asyncio.AbstractEventLoop
implementation that delegates time-based scheduling to the Rust simulator,
enabling deterministic testing of Python async code.
"""

import asyncio
import random
from typing import Any, Callable, Dict, Optional

from simulator_py_ext import SimulatorPyExt


class RustBackedEventLoop(asyncio.AbstractEventLoop):
    """
    asyncio event loop backed by the Rust deterministic simulator.

    This event loop delegates all time-based scheduling to the Rust simulator,
    enabling deterministic testing of Python async code alongside Rust entities.

    - Scheduling goes to Rust via simulator.schedule_callback()
    - Call run() to execute all scheduled tasks
    - Callbacks are stored by handle_id; Rust calls _fire_handle() when ready
    """

    def __init__(self, rust_simulator: SimulatorPyExt) -> None:
        super().__init__()
        self._rust_simulator = rust_simulator
        self._callbacks: Dict[int, Callable[[], None]] = {}
        self._next_handle_id: int = 0
        self._running: bool = False
        self._closed: bool = False

    def _seconds_to_ns(self, seconds: float) -> int:
        """Convert float seconds to int nanoseconds."""
        return int(seconds * 1_000_000_000)

    def _ns_to_seconds(self, nanoseconds: int) -> float:
        """Convert int nanoseconds to float seconds."""
        return nanoseconds / 1_000_000_000

    def time(self) -> float:
        """Return the current virtual time in seconds."""
        return self._ns_to_seconds(self._rust_simulator.current_time_ns())

    def call_later(
        self,
        delay: float,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[Any] = None,
    ) -> asyncio.TimerHandle:
        """Schedule callback to be called after the given delay in virtual time."""
        delay_ns = self._seconds_to_ns(delay)
        when_ns = self._rust_simulator.current_time_ns() + delay_ns
        when_seconds = self._ns_to_seconds(when_ns)

        handle = asyncio.TimerHandle(when_seconds, callback, args, self, context)

        handle_id = self._next_handle_id
        self._next_handle_id += 1
        self._callbacks[handle_id] = handle._run

        self._rust_simulator.schedule_callback(delay_ns, handle_id, self)

        return handle

    def call_soon(
        self,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[Any] = None,
    ) -> asyncio.TimerHandle:
        """Schedule callback to be called as soon as possible (delay=0)."""
        return self.call_later(0, callback, *args, context=context)

    def call_at(
        self,
        when: float,
        callback: Callable[..., Any],
        *args: Any,
        context: Optional[Any] = None,
    ) -> asyncio.TimerHandle:
        """Schedule callback to be called at the given virtual time."""
        delay = when - self.time()
        return self.call_later(max(0, delay), callback, *args, context=context)

    def _fire_handle(self, handle_id: int) -> None:
        """Called by Rust when a scheduled timer fires."""
        # Set this loop as the running loop so asyncio primitives work correctly
        old_loop = None
        try:
            old_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        asyncio._set_running_loop(self)  # pyre-ignore[16]

        try:
            callback = self._callbacks.pop(handle_id, None)
            if callback is not None:
                callback()
        finally:
            asyncio._set_running_loop(old_loop)  # pyre-ignore[16]

    def run(self) -> None:
        """Run the event loop until all tasks complete."""
        if self._running:
            raise RuntimeError("Event loop is already running")

        self._running = True

        old_loop = None
        try:
            old_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        asyncio._set_running_loop(self)  # pyre-ignore[16]

        try:
            self._rust_simulator.run()
        finally:
            self._running = False
            asyncio._set_running_loop(old_loop)  # pyre-ignore[16]

    def run_forever(self) -> None:
        """Run the event loop until all tasks complete (alias for run())."""
        self.run()

    def run_until_complete(self, future: Any) -> Any:
        """Run until the given future completes and return its result."""
        if self._running:
            raise RuntimeError("Event loop is already running")

        future = asyncio.ensure_future(future, loop=self)

        self._running = True

        old_loop = None
        try:
            old_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        asyncio._set_running_loop(self)  # pyre-ignore[16]

        try:
            self._rust_simulator.run()

            if future.done():
                return future.result()
            else:
                future.cancel()
                raise asyncio.CancelledError()
        finally:
            self._running = False
            asyncio._set_running_loop(old_loop)  # pyre-ignore[16]

    def is_running(self) -> bool:
        """Return True if the event loop is currently running."""
        return self._running

    def is_closed(self) -> bool:
        """Return True if the event loop is closed."""
        return self._closed

    def close(self) -> None:
        """Close the event loop."""
        if self._running:
            raise RuntimeError("Cannot close a running event loop")
        if not self._closed:
            self._closed = True
            self._callbacks.clear()

    def create_task(
        self,
        coro: Any,
        *,
        name: Optional[str] = None,
        context: Optional[Any] = None,
    ) -> asyncio.Task:  # pyre-ignore[15]
        """Create a Task from a coroutine."""
        return asyncio.Task(coro, loop=self, name=name, context=context)

    def create_future(self) -> asyncio.Future:  # pyre-ignore[15]
        """Create a Future associated with this event loop."""
        return asyncio.Future(loop=self)

    def get_debug(self) -> bool:
        """Get the debug mode of the event loop."""
        return False

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug mode (no-op)."""
        pass

    def default_exception_handler(self, context: Dict[str, Any]) -> None:
        """Default exception handler prints context."""
        print(f"[RustBackedEventLoop] Exception: {context}")

    def call_exception_handler(self, context: Dict[str, Any]) -> None:
        """Call the exception handler with the given context."""
        self.default_exception_handler(context)

    def _timer_handle_cancelled(self, handle: asyncio.TimerHandle) -> None:
        """Internal method called when a timer handle is cancelled."""
        pass

    async def shutdown_asyncgens(self) -> None:
        """Shutdown all active async generators."""
        pass

    async def shutdown_default_executor(self, timeout: Optional[float] = None) -> None:
        """Shutdown the default executor."""
        pass

    def __getattr__(self, name: str) -> Any:
        """Raise NotImplementedError for unimplemented AbstractEventLoop methods."""

        def _not_implemented(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                f"{name} is not supported in RustBackedEventLoop. "
                f"This is a minimal event loop for deterministic testing."
            )

        return _not_implemented


class RustBackedEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    """
    Event loop policy that creates RustBackedEventLoop instances.

    Each call to new_event_loop() creates a fresh SimulatorPyExt with
    a random seed, enabling use with IsolatedAsyncioTestCase.
    """

    def new_event_loop(self) -> RustBackedEventLoop:
        """Create a new RustBackedEventLoop with a fresh simulator."""
        seed = random.randint(0, 2**31 - 1)
        simulator = SimulatorPyExt(seed)
        return RustBackedEventLoop(simulator)  # pyre-ignore[45]
