# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Port allocation for agent HTTP servers."""

import asyncio


class PortAllocator:
    """Simple port allocator for agent HTTP servers."""

    def __init__(self, start: int = 9000, end: int = 9999) -> None:
        self.start = start
        self.end = end
        self._used: set[int] = set()
        self._lock = asyncio.Lock()

    async def allocate(self) -> int:
        """Allocate the next available port. Raises if exhausted."""
        async with self._lock:
            for port in range(self.start, self.end):
                if port not in self._used:
                    self._used.add(port)
                    return port
            raise RuntimeError(f"No ports available in range {self.start}-{self.end}")

    async def release(self, port: int) -> None:
        """Release a previously allocated port."""
        async with self._lock:
            self._used.discard(port)
