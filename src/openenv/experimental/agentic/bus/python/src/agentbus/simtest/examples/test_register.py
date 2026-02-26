# pyre-strict
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Unit tests demonstrating deterministic simulation testing.

Shows how to use RustBackedEventLoop to write deterministic tests for
concurrent async code, including tests that reliably detect race conditions.
"""

import asyncio
import unittest

from agentbus.simtest import RustBackedSimulatorTestCase
from agentbus.simtest.examples import (
    AbstractRegister,
    BigFatLockRegister,
    BuggyRegister,
)
from parameterized import parameterized


class TestRegister(RustBackedSimulatorTestCase):
    """Parameterized tests for register implementations using simulation framework."""

    @parameterized.expand(
        [
            ("BigFatLockRegister", BigFatLockRegister),
            ("BuggyRegister", BuggyRegister),
        ]
    )
    async def test_register_sequential(
        self, name: str, register_class: type[AbstractRegister]
    ) -> None:
        """Test that sequential operations work correctly for all implementations."""
        # pyre-ignore[45]: Pyre doesn't know this is a concrete subclass
        register = register_class(rng=self._rng, initial_value=0)

        # Sequential writes
        for i in range(100):
            await register.write(i)

        read_value = await register.read()
        self.assertEqual(read_value, 99)

        # Sequential increments
        for _ in range(100):
            await register.increment()

        read_value = await register.read()
        self.assertEqual(read_value, 199)

    async def _run_workload(self, reg: AbstractRegister, iterations: int = 100) -> None:
        """Helper: Run a simple increment workload."""
        for _ in range(iterations):
            await reg.increment()

    async def _do_concurrent_work(
        self, reg: AbstractRegister, num_tasks: int = 3, iterations: int = 100
    ) -> int:
        """Helper: Run concurrent increment workload and return final value."""
        tasks = [
            asyncio.create_task(self._run_workload(reg, iterations))
            for _ in range(num_tasks)
        ]
        await asyncio.gather(*tasks)
        return await reg.read()

    @parameterized.expand([("BigFatLockRegister", BigFatLockRegister)])
    async def test_register_concurrent_correct(
        self, name: str, register_class: type[AbstractRegister]
    ) -> None:
        """Test that concurrent increments work correctly with proper locking."""
        # pyre-ignore[45]: Pyre doesn't know this is a concrete subclass
        register = register_class(rng=self._rng, initial_value=0)
        final_value = await self._do_concurrent_work(
            register, num_tasks=3, iterations=100
        )

        # With 3 tasks doing 100 increments each, we expect 300
        self.assertEqual(final_value, 300)

    @parameterized.expand([("BuggyRegister", BuggyRegister)])
    async def test_register_concurrent_buggy(
        self, name: str, register_class: type[AbstractRegister]
    ) -> None:
        """Test that concurrent increments fail predictably without locking."""
        # pyre-ignore[45]: Pyre doesn't know this is a concrete subclass
        register = register_class(rng=self._rng, initial_value=0)
        final_value = await self._do_concurrent_work(
            register, num_tasks=3, iterations=100
        )

        # Without proper locking, we expect lost updates (value < 300)
        self.assertLess(
            final_value,
            300,
            f"Expected race condition, but got correct value {final_value}",
        )


if __name__ == "__main__":
    unittest.main()
