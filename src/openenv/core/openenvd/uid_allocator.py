# SPDX-License-Identifier: BSD-3-Clause

"""Dedicated-UID allocation for supervised tasks.

Every task registered under a privileged daemon gets its own UID/GID pair
by default, so sibling tasks are mutually blind: they cannot signal each
other, read each other's files, or tamper with another task's outputs.
Explicit ``uid``/``gid`` on a TaskSpec always wins; ``auto_uid=False``
opts out entirely.
"""

from __future__ import annotations

from typing import Optional, Tuple


DEFAULT_UID_BASE = 65536
DEFAULT_UID_COUNT = 4096


class UidAllocator:
    """Allocates the lowest free UID pair in a configured range."""

    def __init__(self, base: int = DEFAULT_UID_BASE, count: int = DEFAULT_UID_COUNT):
        if count <= 0:
            raise ValueError("count must be positive")
        self._base = base
        self._count = count
        self._in_use: dict[str, int] = {}
        self._free = set(range(base, base + count))

    def acquire(self, name: str) -> Optional[Tuple[int, int]]:
        """Return a free ``(uid, gid)`` for ``name``, or None if exhausted."""
        if name in self._in_use:
            uid = self._in_use[name]
            return uid, uid
        try:
            uid = min(self._free)
        except ValueError:
            return None
        self._free.discard(uid)
        self._in_use[name] = uid
        return uid, uid

    def release(self, name: str) -> None:
        uid = self._in_use.pop(name, None)
        if uid is not None:
            self._free.add(uid)

    @property
    def available(self) -> int:
        return len(self._free)
