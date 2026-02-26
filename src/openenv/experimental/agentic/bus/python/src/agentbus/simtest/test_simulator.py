# pyre-strict
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Unit tests for RustBackedEventLoop.

Tests verify sleep, determinism, and concurrent task execution.
"""

import asyncio
import random
import unittest
from typing import Callable

from agentbus.simtest import RustBackedEventLoop, RustBackedSimulatorTestCase
from simulator_py_ext import SimulatorPyExt


class TestSimulatorDeterminism(unittest.TestCase):
    """Test that simulator event loops produce deterministic results."""

    NUM_ITERATIONS = 1000

    def _run_two_tasks(self, loop: asyncio.AbstractEventLoop) -> int:
        """Run 2 tasks with near-identical sleep times, return which finished first."""
        completion_order: list[int] = []

        async def task_worker(task_id: int, sleep_duration: float) -> None:
            await asyncio.sleep(sleep_duration)
            completion_order.append(task_id)

        async def main() -> None:
            # Task 1 has slightly LONGER sleep than Task 2
            # Task 2 should finish first, but jitter might flip the order
            t1 = asyncio.create_task(task_worker(1, 0.00101))  # 1.01ms
            t2 = asyncio.create_task(task_worker(2, 0.001))  # 1.00ms
            await asyncio.gather(t1, t2)

        loop.run_until_complete(main())
        loop.close()
        return completion_order[0]

    def _assert_determinism(
        self,
        make_loop: Callable[[], asyncio.AbstractEventLoop],
        name: str,
    ) -> None:
        """Assert that a loop type produces deterministic results."""
        results = [self._run_two_tasks(make_loop()) for _ in range(self.NUM_ITERATIONS)]
        count_1_first = results.count(1)
        self.assertEqual(
            count_1_first,
            0,
            f"{name} should be deterministic: Task 1 won {count_1_first}/{self.NUM_ITERATIONS}",
        )

    def test_rust_backed_event_loop_determinism(self) -> None:
        """Test that RustBackedEventLoop produces identical results."""
        self._assert_determinism(
            # pyre-ignore[45]
            lambda: RustBackedEventLoop(SimulatorPyExt(random.randint(0, 2**63 - 1))),
            "RustBackedEventLoop",
        )

    def test_default_loop_nondeterminism(self) -> None:
        """Demonstrate that the default asyncio loop may be non-deterministic."""

        # Just here to document non-determinism. Will likely fail if uncommented.
        # self._assert_determinism(asyncio.new_event_loop, "Default loop")


class TestRustBackedEventLoop(RustBackedSimulatorTestCase):
    """Test RustBackedEventLoop functionality."""

    async def test_sleep_advances_time(self) -> None:
        """Test that sleep advances virtual time correctly."""
        loop = asyncio.get_running_loop()

        start_time = loop.time()
        sleep_duration = 0.05  # 50ms

        # Sleep should advance virtual time
        await asyncio.sleep(sleep_duration)

        end_time = loop.time()
        elapsed = end_time - start_time

        self.assertGreaterEqual(
            elapsed,
            sleep_duration,
            "Sleep should advance virtual time by at least the requested duration",
        )

    async def test_multiple_concurrent_sleeps(self) -> None:
        """Test that multiple concurrent sleeps are ordered correctly by virtual time."""
        event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

        results: list[tuple[float, int]] = []

        async def sleeper(duration: float, task_id: int) -> None:
            """Sleep for a duration and record completion time."""
            await asyncio.sleep(duration)
            results.append((event_loop.time(), task_id))

        # Spawn multiple tasks with different sleep durations
        tasks = [
            asyncio.create_task(sleeper(0.3, 3)),
            asyncio.create_task(sleeper(0.1, 1)),
            asyncio.create_task(sleeper(0.2, 2)),
        ]

        await asyncio.gather(*tasks)

        # Verify tasks completed in order of their sleep durations
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][1], 1, "Task 1 (0.1s) should complete first")
        self.assertEqual(results[1][1], 2, "Task 2 (0.2s) should complete second")
        self.assertEqual(results[2][1], 3, "Task 3 (0.3s) should complete third")

        # Verify times are correct
        self.assertAlmostEqual(results[0][0], 0.1, places=6)
        self.assertAlmostEqual(results[1][0], 0.2, places=6)
        self.assertAlmostEqual(results[2][0], 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
