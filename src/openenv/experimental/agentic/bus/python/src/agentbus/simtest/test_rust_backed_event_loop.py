# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-unsafe

"""
RustBackedEventLoop tests.

These tests validate that RustBackedEventLoop correctly integrates with the
Rust simulator to drive Python async patterns: coroutines, generators, queues,
futures, and long-running tasks.
"""

import asyncio
import random
import unittest

from agentbus.simtest.rust_backed_event_loop import RustBackedEventLoop
from simulator_py_ext import SimulatorPyExt


class TestRustBackedEventLoop(unittest.TestCase):
    """Tests validating RustBackedEventLoop works with Rust simulator."""

    def setUp(self) -> None:
        """Create simulator and event loop for each test."""
        self._seed = random.randint(0, 2**31 - 1)
        print(f"Test seed: {self._seed}")
        self.simulator = SimulatorPyExt(self._seed)
        # pyre-ignore[45]: RustBackedEventLoop intentionally doesn't implement all abstract methods
        self.loop = RustBackedEventLoop(self.simulator)
        asyncio.set_event_loop(self.loop)

    def test_simple_coro(self) -> None:
        """Test simple coroutine returns value."""

        async def simple() -> int:
            return 42

        task = self.loop.create_task(simple())
        self.loop.run()
        self.assertEqual(task.result(), 42)

    def test_coro_with_await(self) -> None:
        """Test coroutine with internal await."""

        async def compute() -> str:
            await asyncio.sleep(0.001)
            return "done"

        task = self.loop.create_task(compute())
        self.loop.run()
        self.assertEqual(task.result(), "done")

    def test_sleep_advances_time(self) -> None:
        """Test that sleep advances virtual time."""

        async def measure_time() -> float:
            t1 = self.loop.time()
            await asyncio.sleep(1.0)
            t2 = self.loop.time()
            return t2 - t1

        task = self.loop.create_task(measure_time())
        self.loop.run()
        elapsed = task.result()
        self.assertGreaterEqual(elapsed, 1.0)

    def test_async_generator(self) -> None:
        """Test async generator with multiple yields."""

        async def gen():
            yield 1
            await asyncio.sleep(0.001)
            yield 2
            yield 3

        async def collect() -> list:
            result = []
            async for v in gen():
                result.append(v)
            return result

        task = self.loop.create_task(collect())
        self.loop.run()
        self.assertEqual(task.result(), [1, 2, 3])

    def test_long_running_task(self) -> None:
        """Test long-running fire-and-forget task."""
        counter = [0]

        async def worker() -> None:
            while counter[0] < 5:
                await asyncio.sleep(0.001)
                counter[0] += 1

        self.loop.create_task(worker())
        self.loop.run()
        self.assertEqual(counter[0], 5)

    def test_concurrent_tasks(self) -> None:
        """Test multiple concurrent tasks."""
        results = []

        async def task_a() -> None:
            await asyncio.sleep(0.002)
            results.append("a")

        async def task_b() -> None:
            await asyncio.sleep(0.001)
            results.append("b")

        self.loop.create_task(task_a())
        self.loop.create_task(task_b())
        self.loop.run()
        # b finishes first (shorter sleep)
        self.assertEqual(results, ["b", "a"])

    def test_queue_communication(self) -> None:
        """Test queue communication between tasks."""
        queue: asyncio.Queue = asyncio.Queue()
        received = []

        async def producer() -> None:
            for i in range(3):
                await queue.put(i)
                await asyncio.sleep(0.001)

        async def consumer() -> None:
            for _ in range(3):
                item = await queue.get()
                received.append(item)

        self.loop.create_task(producer())
        self.loop.create_task(consumer())
        self.loop.run()
        self.assertEqual(received, [0, 1, 2])

    def test_future_resolution(self) -> None:
        """Test future resolution across tasks."""
        # pyre-ignore[1001]: future is awaited in waiter() coroutine
        future = self.loop.create_future()
        waiter_result = [None]

        async def setter() -> None:
            await asyncio.sleep(0.001)
            future.set_result("resolved")

        async def waiter() -> None:
            result = await future
            waiter_result[0] = result

        self.loop.create_task(setter())
        self.loop.create_task(waiter())
        self.loop.run()
        self.assertEqual(waiter_result[0], "resolved")


if __name__ == "__main__":
    unittest.main()
