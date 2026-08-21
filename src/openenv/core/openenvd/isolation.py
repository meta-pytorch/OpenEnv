# SPDX-License-Identifier: BSD-3-Clause

"""OS-level isolation primitives for openenvd children.

Two mechanisms, applied where the runtime allows:

- UID/GID drop: the child runs as an unprivileged user so it cannot reach
  openenvd-owned resources (control sockets, privileged assets).
- Network namespace: the child gets a fresh netns (loopback only) so
  surfaces bound in the daemon's namespace are structurally unreachable.

Both degrade gracefully: when the capability is missing the child is spawned
without that isolation and a warning is returned to the caller.
"""

from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import socket
import struct
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from openenv.core.openenvd.models import TaskSpec

logger = logging.getLogger(__name__)

CLONE_NEWNET = 0x40000000
IFF_UP = 0x1
SIOCSIFFLAGS = 0x8914


@dataclass(frozen=True)
class IsolationCapabilities:
    """What isolation mechanisms this process can apply to children."""

    can_drop_uid: bool
    can_unshare_net: bool


def detect_capabilities() -> IsolationCapabilities:
    """Probe which isolation mechanisms are available right now."""
    return IsolationCapabilities(
        can_drop_uid=os.geteuid() == 0,
        can_unshare_net=_probe_unshare_net(),
    )


def build_preexec(
    spec: TaskSpec,
    capabilities: Optional[IsolationCapabilities] = None,
) -> Tuple[Optional[Callable[[], None]], list[str]]:
    """Build a ``preexec_fn`` applying the spec's requested isolation.

    Returns ``(preexec, warnings)``. ``preexec`` is ``None`` when no
    isolation applies; warnings describe any requested-but-unavailable
    mechanism.
    """
    caps = capabilities or detect_capabilities()
    warnings: list[str] = []

    need_net = spec.network_isolated
    if need_net and not caps.can_unshare_net:
        warnings.append(
            "network isolation requested but unavailable; "
            "child will share the daemon's network namespace"
        )
        need_net = False

    need_ids = spec.uid is not None or spec.gid is not None
    if need_ids and not caps.can_drop_uid:
        warnings.append(
            "uid/gid drop requested but the daemon is not privileged; "
            "child will run as the current user"
        )
        need_ids = False

    if not (need_net or need_ids):
        return None, warnings

    def preexec() -> None:
        if need_net:
            _unshare_net()
        if need_ids:
            os.setgroups([])
            if spec.gid is not None:
                os.setgid(spec.gid)
            if spec.uid is not None:
                os.setuid(spec.uid)

    return preexec, warnings


def _load_libc() -> Optional[ctypes.CDLL]:
    try:
        return ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        logger.debug("libc unavailable; network namespaces unsupported")
        return None


def _probe_unshare_net() -> bool:
    """Check CLONE_NEWNET availability without moving ourselves into a new ns."""
    libc = _load_libc()
    if libc is None:
        return False
    try:
        pid = os.fork()
    except OSError:
        return False
    if pid == 0:
        try:
            rc = libc.unshare(CLONE_NEWNET)
        except BaseException:
            os._exit(1)
        os._exit(0 if rc == 0 else 1)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status) == 0


def _unshare_net() -> bool:
    """Move the calling process into a fresh network namespace."""
    libc = _load_libc()
    if libc is None:
        return False
    if libc.unshare(CLONE_NEWNET) != 0:
        return False
    _bring_loopback_up()
    return True


def _bring_loopback_up() -> None:
    """Bring ``lo`` up inside a fresh netns (it starts down)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_IP)
    except OSError:
        return
    try:
        ifreq = struct.pack("16sH", b"lo", IFF_UP)
        fcntl.ioctl(s.fileno(), SIOCSIFFLAGS, ifreq)
    except OSError:
        logger.debug("could not bring loopback up in new netns")
    finally:
        s.close()
