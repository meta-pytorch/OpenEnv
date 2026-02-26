# pyre-strict
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Example: Thread-safe register implementations for simulation testing.

Demonstrates testing concurrent code using the simulation framework with both
correct (BigFatLockRegister) and buggy (BuggyRegister) implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from random import Random


class AbstractRegister(ABC):
    """Abstract interface for a register with async read/write operations."""

    @abstractmethod
    def __init__(self, rng: Random, initial_value: int = 0) -> None:
        """
        Initialize the register.

        Args:
            rng: Random number generator for simulating delays
            initial_value: Initial value of the register
        """
        pass

    @abstractmethod
    async def read(self) -> int:
        """Read the current value of the register."""
        pass

    @abstractmethod
    async def write(self, value: int) -> None:
        """Write a new value to the register."""
        pass

    @abstractmethod
    async def increment(self) -> int:
        """Atomically increment the register and return the new value."""
        pass


class BigFatLockRegister(AbstractRegister):
    """Correct implementation using a lock to protect concurrent access."""

    def __init__(self, rng: Random, initial_value: int = 0) -> None:
        """Initialize the register with RNG and initial value."""
        self._value: int = initial_value
        self._lock: asyncio.Lock = asyncio.Lock()
        self._rng: Random = rng

    async def _inject_delay(self) -> None:
        """Simulate random processing delay."""
        await asyncio.sleep(self._rng.randint(1, 10) / 1000.0)

    async def read(self) -> int:
        """Read the register value with proper locking."""
        async with self._lock:
            await self._inject_delay()
            return self._value

    async def write(self, value: int) -> None:
        """Write to the register with proper locking."""
        async with self._lock:
            await self._inject_delay()
            self._value = value

    async def increment(self) -> int:
        """Atomically increment the register with proper locking."""
        async with self._lock:
            await self._inject_delay()
            existing_value = self._value
            await self._inject_delay()
            self._value = existing_value + 1
            return self._value


class BuggyRegister(AbstractRegister):
    """Buggy implementation with race conditions (no locks)."""

    def __init__(self, rng: Random, initial_value: int = 0) -> None:
        """Initialize the register with RNG and initial value."""
        self._value: int = initial_value
        self._rng: Random = rng

    async def _inject_delay(self) -> None:
        """Simulate random processing delay."""
        await asyncio.sleep(self._rng.randint(1, 10) / 1000.0)

    async def read(self) -> int:
        """Read the register value (no locking - potential race condition)."""
        await self._inject_delay()
        return self._value

    async def write(self, value: int) -> None:
        """Write to the register (no locking - potential race condition)."""
        await self._inject_delay()
        self._value = value

    async def increment(self) -> int:
        """Increment the register (race condition - delay between read and write)."""
        await self._inject_delay()
        existing_value = self._value
        await self._inject_delay()  # Race window!
        self._value = existing_value + 1
        return self._value
