# SPDX-License-Identifier: BSD-3-Clause

"""Assign task identities from a deployment-reserved UID/GID range.

This is not a system account allocator: the container deployment must reserve the
configured range exclusively for one daemon. IDs are never reused during this
allocator's lifetime, because files or escaped descendants may outlive a task.
"""

from __future__ import annotations


DEFAULT_UID_BASE = 65536
DEFAULT_UID_COUNT = 4096
MAX_UID = 2**32 - 2  # The all-ones value means "leave unchanged" to setuid/setgid.


class UidAllocator:
    """Assign the lowest unused UID/GID pair from an exclusive reserved range."""

    def __init__(self, base: int = DEFAULT_UID_BASE, count: int = DEFAULT_UID_COUNT):
        if base < 1 or count <= 0 or base + count - 1 > MAX_UID:
            raise ValueError("UID range must contain positive, valid POSIX IDs")
        self._in_use: dict[str, int] = {}
        self._issued: set[int] = set()
        self._free = set(range(base, base + count))

    def acquire(self, name: str) -> tuple[int, int]:
        """Return a stable pair for a task; fail when the reserved range is full."""
        if name in self._in_use:
            uid = self._in_use[name]
        elif self._free:
            uid = min(self._free)
            self.reserve(name, uid)
        else:
            raise ValueError("dedicated UID range exhausted")
        return uid, uid

    def reserve(self, name: str, uid: int) -> None:
        """Reserve an explicit UID so automatic allocation cannot collide with it."""
        if not 1 <= uid <= MAX_UID:
            raise ValueError("UID must be a positive, valid POSIX ID")
        if self._in_use.get(name) == uid:
            return
        if name in self._in_use or uid in self._issued:
            raise ValueError("UID is already assigned or was used by another task")
        self._in_use[name] = uid
        self._issued.add(uid)
        self._free.discard(uid)

    def release(self, name: str) -> None:
        """Forget a task without recycling its potentially still-owned identity."""
        self._in_use.pop(name, None)

    @property
    def available(self) -> int:
        return len(self._free)
