# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Tests for PortAllocator."""

import asyncio

import pytest
from agentic.kernel.core.backends.ports import PortAllocator


class TestPortAllocator:
    @pytest.mark.asyncio
    async def test_allocate_returns_port_in_range(self):
        alloc = PortAllocator(start=9000, end=9010)
        port = await alloc.allocate()
        assert 9000 <= port < 9010

    @pytest.mark.asyncio
    async def test_allocate_sequential(self):
        alloc = PortAllocator(start=9000, end=9010)
        p1 = await alloc.allocate()
        p2 = await alloc.allocate()
        assert p1 != p2
        assert p1 == 9000
        assert p2 == 9001

    @pytest.mark.asyncio
    async def test_release_allows_reuse(self):
        alloc = PortAllocator(start=9000, end=9002)
        p1 = await alloc.allocate()
        await alloc.allocate()

        # Pool exhausted
        with pytest.raises(RuntimeError, match="No ports"):
            await alloc.allocate()

        # Release one
        await alloc.release(p1)

        # Can allocate again
        p3 = await alloc.allocate()
        assert p3 == p1

    @pytest.mark.asyncio
    async def test_exhaust_pool_raises(self):
        alloc = PortAllocator(start=9000, end=9003)
        await alloc.allocate()
        await alloc.allocate()
        await alloc.allocate()
        with pytest.raises(RuntimeError, match="No ports"):
            await alloc.allocate()

    @pytest.mark.asyncio
    async def test_concurrent_allocations(self):
        alloc = PortAllocator(start=9000, end=9100)
        ports = await asyncio.gather(*[alloc.allocate() for _ in range(50)])
        # All unique
        assert len(set(ports)) == 50

    @pytest.mark.asyncio
    async def test_release_nonexistent_is_noop(self):
        alloc = PortAllocator(start=9000, end=9010)
        # Should not raise
        await alloc.release(8888)
