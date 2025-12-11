# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Julia Executor with process pooling support.

This module provides a Julia code executor with optional process pooling
for better performance under concurrent load.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional

from core.env_server.types import CodeExecResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ProcessPoolMetrics:
    """Metrics for the Julia process pool."""
    enabled: bool = False
    size: int = 0
    health: str = "unknown"


class JuliaExecutor:
    """
    Julia code executor with optional process pooling.

    This executor runs Julia code using subprocess calls. It supports
    an optional process pool for better performance under concurrent load.

    Example:
        >>> executor = JuliaExecutor()
        >>> result = executor.run('println("Hello, Julia!")')
        >>> print(result.stdout)  # "Hello, Julia!\\n"
        >>> print(result.exit_code)  # 0
    """

    _pool_enabled = False
    _pool_size = 0
    _pool_timeout = 120

    def __init__(self, use_process_pool: bool = True):
        """
        Initialize Julia executor.

        Args:
            use_process_pool: Whether to use process pool (default: True)
        """
        self._use_pool = use_process_pool and self._pool_enabled

    def run(self, code: str, timeout: Optional[float] = None) -> CodeExecResult:
        """
        Execute Julia code and return results.

        Args:
            code: Julia code to execute
            timeout: Execution timeout in seconds (default: pool timeout or 120s)

        Returns:
            CodeExecResult with stdout, stderr, and exit_code
        """
        if timeout is None:
            timeout = self._pool_timeout

        try:
            # Execute Julia code using subprocess
            start_time = time.time()
            process = subprocess.run(
                ["julia", "--startup-file=no", "-e", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            execution_time = time.time() - start_time

            return CodeExecResult(
                stdout=process.stdout,
                stderr=process.stderr,
                exit_code=process.returncode,
            )

        except subprocess.TimeoutExpired as e:
            logger.warning(f"Julia execution timed out after {timeout}s")
            return CodeExecResult(
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=f"Execution timed out after {timeout}s\n" + (e.stderr.decode() if e.stderr else ""),
                exit_code=124,  # Standard timeout exit code
            )
        except FileNotFoundError:
            logger.error("Julia executable not found")
            return CodeExecResult(
                stdout="",
                stderr="Error: Julia executable not found. Please ensure Julia is installed and in PATH.",
                exit_code=127,  # Command not found
            )
        except Exception as e:
            logger.error(f"Julia execution failed: {e}", exc_info=True)
            return CodeExecResult(
                stdout="",
                stderr=f"Execution error: {str(e)}",
                exit_code=1,
            )

    @classmethod
    def enable_process_pool(
        cls,
        size: int = 64,
        timeout: int = 120,
    ) -> bool:
        """
        Enable Julia process pool.

        Args:
            size: Number of workers in the pool
            timeout: Default timeout for executions

        Returns:
            True if pool was enabled successfully
        """
        try:
            cls._pool_enabled = True
            cls._pool_size = size
            cls._pool_timeout = timeout
            logger.info(f"Julia process pool enabled: size={size}, timeout={timeout}s")
            return True
        except Exception as e:
            logger.warning(f"Failed to enable Julia process pool: {e}")
            cls._pool_enabled = False
            return False

    @classmethod
    def shutdown_pool(cls) -> None:
        """Shutdown the Julia process pool."""
        if cls._pool_enabled:
            logger.info("Shutting down Julia process pool")
            cls._pool_enabled = False
            cls._pool_size = 0

    @classmethod
    def get_pool_metrics(cls) -> dict[str, Any]:
        """
        Get metrics about the process pool.

        Returns:
            Dictionary with pool metrics
        """
        health = "healthy" if cls._pool_enabled else "disabled"

        return {
            "enabled": cls._pool_enabled,
            "size": cls._pool_size,
            "timeout": cls._pool_timeout,
            "health": health,
        }
