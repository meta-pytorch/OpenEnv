# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Emulator Pool Manager for parallel training.

This module provides a pool of pre-warmed Android emulators for
high-throughput parallel training on multi-core systems.
"""

import logging
import queue
import threading
import time
from typing import Dict, List, Optional

from .android_environment import AndroidEnvironment

logger = logging.getLogger(__name__)


class EmulatorPool:
    """
    Pool of pre-warmed Android emulators for parallel training.

    The pool:
    1. Boots N emulators at startup (amortizes 30-60s boot time)
    2. Keeps emulators running across episodes
    3. Resets app state (not full emulator) between episodes
    4. Provides instant environment access via get/put

    Optimized for systems with 100+ CPU cores and high memory capacity.

    Example:
        >>> # Boot 64 emulators once at startup (10 min one-time cost)
        >>> pool = EmulatorPool(
        ...     pool_size=64,
        ...     task_path="/workspace/tasks/my_task.textproto",
        ...     avd_name="default_pixel_6"
        ... )
        >>>
        >>> # Training loop - instant access!
        >>> for episode in range(10000):
        ...     env = pool.get()  # <1ms
        ...     # ... run episode ...
        ...     pool.put(env)  # Returns env to pool (resets app state)
        >>>
        >>> pool.close()
    """

    def __init__(
        self,
        pool_size: int,
        task_path: str,
        avd_name: str,
        adb_path: str = "~/Android/Sdk/platform-tools/adb",
        emulator_path: str = "~/Android/Sdk/emulator/emulator",
        android_avd_home: str = "~/.android/avd",
        android_sdk_root: str = "~/Android/Sdk",
        run_headless: bool = True,
        image_format: str = "JPEG",
        image_quality: int = 85,
        use_shared_memory: bool = False,
    ):
        """Initialize emulator pool.

        Args:
            pool_size: Number of emulators to pre-warm.
            task_path: Path to task textproto.
            avd_name: Name of Android Virtual Device.
            adb_path: Path to ADB executable.
            emulator_path: Path to emulator executable.
            android_avd_home: AVD home directory.
            android_sdk_root: SDK root directory.
            run_headless: Run emulators headless.
            image_format: Image encoding format.
            image_quality: JPEG quality (1-100).
            use_shared_memory: Use shared memory optimization.
        """
        self.pool_size = pool_size
        self.task_path = task_path
        self.avd_name = avd_name
        self.adb_path = adb_path
        self.emulator_path = emulator_path
        self.android_avd_home = android_avd_home
        self.android_sdk_root = android_sdk_root
        self.run_headless = run_headless
        self.image_format = image_format
        self.image_quality = image_quality
        self.use_shared_memory = use_shared_memory

        # Thread-safe queue for available emulators
        self._available: queue.Queue = queue.Queue(maxsize=pool_size)
        self._all_emulators: List[AndroidEnvironment] = []
        self._lock = threading.Lock()
        self._closed = False

        # Boot all emulators
        logger.info(f"Booting {pool_size} emulators... (this will take ~{pool_size} minutes)")
        self._boot_pool()
        logger.info(f"Emulator pool ready with {pool_size} instances!")

    def _boot_pool(self):
        """Boot all emulators in the pool."""
        start_time = time.time()

        for i in range(self.pool_size):
            logger.info(f"Booting emulator {i+1}/{self.pool_size}...")

            # Create unique shared memory name if using shared memory
            shm_name = f"android_pool_{i}" if self.use_shared_memory else None

            env = AndroidEnvironment(
                task_path=self.task_path,
                avd_name=self.avd_name,
                adb_path=self.adb_path,
                emulator_path=self.emulator_path,
                android_avd_home=self.android_avd_home,
                android_sdk_root=self.android_sdk_root,
                run_headless=self.run_headless,
                image_format=self.image_format,
                image_quality=self.image_quality,
                use_shared_memory=self.use_shared_memory,
                shared_memory_name=shm_name,
            )

            # Reset to ensure ready state
            env.reset()

            self._all_emulators.append(env)
            self._available.put(env)

        elapsed = time.time() - start_time
        logger.info(f"Pool boot complete in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"Average boot time per emulator: {elapsed/self.pool_size:.1f} seconds")

    def get(self, timeout: Optional[float] = None) -> AndroidEnvironment:
        """Get an emulator from the pool.

        Args:
            timeout: Max time to wait for available emulator (seconds).
                    None = wait forever.

        Returns:
            AndroidEnvironment ready for use.

        Raises:
            queue.Empty: If timeout expires and no emulator available.
            RuntimeError: If pool is closed.
        """
        if self._closed:
            raise RuntimeError("Emulator pool is closed")

        try:
            env = self._available.get(timeout=timeout)
            logger.debug(f"Dispatched emulator from pool ({self._available.qsize()} remaining)")
            return env
        except queue.Empty:
            raise queue.Empty(
                f"No emulator available after {timeout}s. "
                f"Pool size={self.pool_size}, all in use."
            )

    def put(self, env: AndroidEnvironment, reset: bool = True):
        """Return an emulator to the pool.

        Args:
            env: Environment to return.
            reset: Whether to reset the environment before returning to pool.
                  Set to False if you've already reset it.
        """
        if self._closed:
            logger.warning("Attempted to return emulator to closed pool")
            return

        if reset:
            # Fast reset: just reset app state, not full emulator
            # This takes ~1s vs 30-60s for full emulator boot
            try:
                env.reset()
            except Exception as e:
                logger.error(f"Error resetting emulator: {e}")
                # Still return to pool, it might recover

        self._available.put(env)
        logger.debug(f"Returned emulator to pool ({self._available.qsize()} available)")

    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics.

        Returns:
            Dict with pool_size, available, in_use counts.
        """
        available = self._available.qsize()
        return {
            "pool_size": self.pool_size,
            "available": available,
            "in_use": self.pool_size - available,
        }

    def close(self):
        """Close all emulators in the pool."""
        if self._closed:
            return

        logger.info("Closing emulator pool...")
        self._closed = True

        # Close all emulators
        for env in self._all_emulators:
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing emulator: {e}")

        logger.info("Emulator pool closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class EmulatorPoolManager:
    """
    Manager for multiple emulator pools (for multi-task training).

    Allows running multiple tasks simultaneously with separate pools.

    Example:
        >>> manager = EmulatorPoolManager()
        >>> manager.create_pool("task1", pool_size=32, task_path="/tasks/task1.textproto", ...)
        >>> manager.create_pool("task2", pool_size=32, task_path="/tasks/task2.textproto", ...)
        >>>
        >>> # Get emulator for specific task
        >>> env = manager.get("task1")
        >>> # ... use env ...
        >>> manager.put("task1", env)
    """

    def __init__(self):
        """Initialize the pool manager."""
        self._pools: Dict[str, EmulatorPool] = {}
        self._lock = threading.Lock()

    def create_pool(self, name: str, **pool_kwargs) -> EmulatorPool:
        """Create a new emulator pool.

        Args:
            name: Unique name for this pool.
            **pool_kwargs: Arguments passed to EmulatorPool constructor.

        Returns:
            Created EmulatorPool.
        """
        with self._lock:
            if name in self._pools:
                raise ValueError(f"Pool '{name}' already exists")

            pool = EmulatorPool(**pool_kwargs)
            self._pools[name] = pool
            logger.info(f"Created pool '{name}' with {pool.pool_size} emulators")
            return pool

    def get(self, pool_name: str, timeout: Optional[float] = None) -> AndroidEnvironment:
        """Get emulator from named pool."""
        pool = self._pools.get(pool_name)
        if not pool:
            raise ValueError(f"Pool '{pool_name}' not found")
        return pool.get(timeout=timeout)

    def put(self, pool_name: str, env: AndroidEnvironment, reset: bool = True):
        """Return emulator to named pool."""
        pool = self._pools.get(pool_name)
        if not pool:
            raise ValueError(f"Pool '{pool_name}' not found")
        pool.put(env, reset=reset)

    def get_stats(self, pool_name: Optional[str] = None) -> Dict:
        """Get statistics for one or all pools."""
        if pool_name:
            pool = self._pools.get(pool_name)
            if not pool:
                raise ValueError(f"Pool '{pool_name}' not found")
            return {pool_name: pool.get_stats()}
        else:
            return {name: pool.get_stats() for name, pool in self._pools.items()}

    def close(self, pool_name: Optional[str] = None):
        """Close one or all pools."""
        if pool_name:
            pool = self._pools.pop(pool_name, None)
            if pool:
                pool.close()
        else:
            for pool in self._pools.values():
                pool.close()
            self._pools.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
