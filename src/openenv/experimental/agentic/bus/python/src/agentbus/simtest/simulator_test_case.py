# pyre-strict
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Base test class for simulation testing.

Provides RustBackedSimulatorTestCase which handles event loop policy
setup/teardown and common patterns for writing deterministic async tests.
"""

import asyncio
import random
import unittest

from agentbus.simtest.rust_backed_event_loop import (
    RustBackedEventLoop,
    RustBackedEventLoopPolicy,
)


class RustBackedSimulatorTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Base class for tests using RustBackedEventLoop.

    Automatically configures RustBackedEventLoopPolicy for the test class
    and provides helper methods for common simulation testing patterns.

    An RNG with a random seed is automatically set up for each test and the
    seed is printed for reproducibility.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Set up custom event loop policy for the entire test class."""
        cls._original_policy = asyncio.get_event_loop_policy()
        asyncio.set_event_loop_policy(RustBackedEventLoopPolicy())

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore original event loop policy."""
        asyncio.set_event_loop_policy(cls._original_policy)

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        super().setUp()
        self._seed = random.randint(0, 2**31 - 1)
        print(f"Test seed: {self._seed}")
        self._rng = random.Random(self._seed)

    def get_event_loop(self) -> RustBackedEventLoop:
        """
        Get the current event loop as a RustBackedEventLoop with type checking.

        Returns:
            The current RustBackedEventLoop instance

        Raises:
            AssertionError: If the current loop is not a RustBackedEventLoop
        """
        loop = asyncio.get_running_loop()
        self.assertIsInstance(
            loop,
            RustBackedEventLoop,
            f"Expected RustBackedEventLoop, got {type(loop)}",
        )
        # Type narrowing for Pyre
        assert isinstance(loop, RustBackedEventLoop)
        return loop
