# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed process isolation without Python callbacks between fork and exec.

Linux setup runs in a fresh, isolated Python interpreter. A private socket carries
the target environment and reports setup failures; target code runs only after
the requested isolation is established. This file also serves as that standalone
helper, so its runtime imports must remain in the standard library.
"""

from __future__ import annotations

import asyncio
import ctypes
import errno
import fcntl
import json
import os
import signal
import socket
import struct
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openenv.core.openenvd.models import TaskSpec


CLONE_NEWNET = 0x40000000
PR_SET_NO_NEW_PRIVS = 38
IFF_UP = 0x1
SIOCSIFFLAGS = 0x8914
_HELPER_TIMEOUT_S = 10.0
_READY = b"ready\n"


class IsolationError(OSError):
    """Requested isolation or process setup could not be established."""


@dataclass(frozen=True)
class IsolationCapabilities:
    """Isolation mechanisms available to this daemon."""

    can_drop_uid: bool
    can_unshare_net: bool
    can_allocate_uids: bool = False


def in_initial_user_ns() -> bool:
    """Whether Linux maps the full UID range into the initial user namespace."""
    try:
        with open("/proc/self/uid_map") as stream:
            return stream.read().split() == ["0", "0", "4294967295"]
    except OSError:
        return False


def detect_capabilities() -> IsolationCapabilities:
    """Probe in a separate executable, never by running Python after fork."""
    return IsolationCapabilities(
        can_drop_uid=os.geteuid() == 0,
        can_unshare_net=_probe_unshare_net(),
        can_allocate_uids=os.geteuid() == 0 and in_initial_user_ns(),
    )


def validate_isolation(spec: TaskSpec, capabilities: IsolationCapabilities) -> None:
    """Reject requested isolation that cannot be applied."""
    if spec.uid is not None or spec.gid is not None:
        if spec.uid is None or spec.gid is None or spec.uid <= 0 or spec.gid <= 0:
            raise IsolationError("isolation requires both a nonzero UID and GID")
        if not capabilities.can_drop_uid:
            raise IsolationError("UID/GID isolation requires a privileged daemon")
    if spec.network_isolated:
        if not capabilities.can_unshare_net or sys.platform != "linux":
            raise IsolationError("network namespace isolation is unavailable")
        if spec.uid is None:
            raise IsolationError("network isolation requires a dedicated UID and GID")


async def spawn_task(
    spec: TaskSpec,
    capabilities: IsolationCapabilities,
    *,
    env: dict[str, str],
    stdout=None,
    stderr=None,
) -> asyncio.subprocess.Process:
    """Spawn a task only after isolation succeeds, with private files and stdin.

    The helper starts with an empty environment and a trusted working directory.
    It receives the task's environment through a private socket, applies it only
    to the final executable, and closes the socket on exec. A failed exec is
    reported to the caller just like failed namespace or credential setup.
    """
    validate_isolation(spec, capabilities)
    options = {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": stdout,
        "stderr": stderr,
        "start_new_session": True,
        "umask": 0o077,
    }
    # Native subprocess options are safe in a daemon with active threads.
    if spec.uid is not None and not spec.network_isolated:
        options.update(user=spec.uid, group=spec.gid, extra_groups=())
    if sys.platform != "linux":
        return await asyncio.create_subprocess_exec(
            *spec.argv, env=env, cwd=spec.cwd or "/", **options
        )

    config = json.dumps(
        {
            "argv": spec.argv,
            "env": env,
            "cwd": spec.cwd,
            "network_isolated": spec.network_isolated,
            "uid": spec.uid if spec.network_isolated else None,
            "gid": spec.gid if spec.network_isolated else None,
        }
    ).encode()
    parent, child = socket.socketpair()
    proc = None
    try:
        parent.setblocking(False)
        proc = await asyncio.create_subprocess_exec(
            *_helper_command(),
            "--spawn",
            str(child.fileno()),
            pass_fds=(child.fileno(),),
            env={},
            cwd="/",
            **options,
        )
        child.close()
        try:
            await asyncio.wait_for(_exchange_setup(parent, config), _HELPER_TIMEOUT_S)
        except asyncio.TimeoutError as error:
            raise IsolationError("isolation helper setup timed out") from error
        return proc
    except BaseException:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
        raise
    finally:
        parent.close()
        child.close()


async def _exchange_setup(channel: socket.socket, config: bytes) -> None:
    loop = asyncio.get_running_loop()
    try:
        await loop.sock_sendall(channel, config)
        channel.shutdown(socket.SHUT_WR)
        response = b""
        while chunk := await loop.sock_recv(channel, 1024):
            response += chunk
            if len(response) > 1024:
                raise IsolationError("invalid isolation helper response")
    except (BrokenPipeError, ConnectionResetError) as error:
        raise IsolationError("isolation helper exited before setup") from error
    if response != _READY:
        detail = response.removeprefix(_READY).decode(errors="replace").strip()
        raise IsolationError(detail or "isolation helper exited before setup")


def _helper_command() -> list[str]:
    # -I ignores PYTHON* and cwd; -S also skips site hooks and .pth files.
    return [sys.executable, "-I", "-S", os.path.abspath(__file__)]


def _probe_unshare_net() -> bool:
    if sys.platform != "linux":
        return False
    try:
        result = subprocess.run(
            [*_helper_command(), "--probe-network"],
            env={},
            cwd="/",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_HELPER_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _unshare_net() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.unshare.argtypes = [ctypes.c_int]
    libc.unshare.restype = ctypes.c_int
    if libc.unshare(CLONE_NEWNET) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
        fcntl.ioctl(channel.fileno(), SIOCSIFFLAGS, struct.pack("16sH", b"lo", IFF_UP))


def _no_new_privileges() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _run_helper(fd: int) -> int:
    with socket.socket(fileno=fd) as channel:
        # pass_fds made this inheritable. EOF after our ready message means that
        # exec succeeded; neither the command nor its descendants receive it.
        channel.set_inheritable(False)
        stage = "read task configuration"
        try:
            with channel.makefile("rb") as stream:
                config = json.load(stream)
            if config["network_isolated"]:
                stage = "create network namespace"
                _unshare_net()
            stage = "disable privilege escalation"
            _no_new_privileges()
            if config["uid"] is not None:
                stage = "drop task credentials"
                os.setgroups([])
                os.setgid(config["gid"])
                os.setuid(config["uid"])
            if config["cwd"] is not None:
                stage = "change task working directory"
                os.chdir(config["cwd"])
            stage = "execute task"
            channel.sendall(_READY)
            os.execvpe(config["argv"][0], config["argv"], config["env"])
        except Exception as error:
            code = getattr(error, "errno", None) or errno.EINVAL
            channel.sendall(f"could not {stage}: {os.strerror(code)}".encode())
    return 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe-network"]:
        try:
            _unshare_net()
        except (OSError, AttributeError):
            sys.exit(1)
    elif len(sys.argv) == 3 and sys.argv[1] == "--spawn":
        sys.exit(_run_helper(int(sys.argv[2])))
    else:
        sys.exit("isolation.py is an internal openenvd helper")
