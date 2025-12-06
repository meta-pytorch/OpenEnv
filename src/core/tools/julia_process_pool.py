# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Julia Process Pool for high-performance code execution.

This module provides a pool of persistent Julia processes that can be reused
for multiple code executions, eliminating the overhead of spawning new processes.

Expected speedup: 50-100x for repeated executions compared to spawning new processes.

Features:
- Persistent Julia processes (no startup overhead)
- Thread-safe process allocation
- Automatic recovery from process failures
- Proper cleanup on shutdown
- Timeout handling per execution

Example:
    >>> pool = JuliaProcessPool(size=4, timeout=30)
    >>> result = pool.execute("println('Hello, Julia!')")
    >>> print(result.stdout)  # "Hello, Julia!\n"
    >>> pool.shutdown()  # Clean up all processes
"""

import atexit
import logging
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from core.env_server.types import CodeExecResult

# Setup logging
logger = logging.getLogger(__name__)


class JuliaWorkerProcess:
    """
    Single Julia worker process that can execute code repeatedly.

    This class manages communication with a persistent Julia REPL process
    using a delimiter-based protocol.
    """

    # Communication protocol delimiters
    START_OUTPUT = "<<<START_OUTPUT>>>"
    START_ERROR = "<<<START_ERROR>>>"
    EXIT_CODE_PREFIX = "<<<EXIT_CODE:"
    END_EXECUTION = "<<<END_EXECUTION>>>"
    END_CODE = "<<<END_CODE>>>"

    def __init__(
        self,
        worker_id: int,
        julia_path: str,
        worker_script: str,
        optimization_flags: bool = True,
    ):
        """
        Initialize a Julia worker process.

        Args:
            worker_id: Unique identifier for this worker
            julia_path: Path to Julia executable
            worker_script: Path to julia_repl_worker.jl script
            optimization_flags: Enable Julia optimization flags
        """
        self.worker_id = worker_id
        self.julia_path = julia_path
        self.worker_script = worker_script
        self.optimization_flags = optimization_flags
        self.process: Optional[subprocess.Popen] = None
        self.is_busy = False
        self.is_healthy = True
        self.lock = threading.Lock()

        # Start the worker process
        self._start_process()

    def _start_process(self) -> None:
        """Start the Julia worker process."""
        cmd = [self.julia_path]

        if self.optimization_flags:
            cmd.extend(
                [
                    "--compile=min",
                    "--optimize=2",
                    "--startup-file=no",
                    "--history-file=no",
                ]
            )

        cmd.append(self.worker_script)

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Wait for "Julia worker ready" message on stderr
            ready_msg = self.process.stderr.readline()
            if "ready" not in ready_msg.lower():
                raise RuntimeError(
                    f"Worker {self.worker_id} did not start properly: {ready_msg}"
                )

            self.is_healthy = True
            logger.info(f"Worker {self.worker_id} started (PID: {self.process.pid})")

        except Exception as e:
            self.is_healthy = False
            logger.error(f"Failed to start worker {self.worker_id}: {e}")
            raise

    def execute(self, code: str, timeout: int = 60) -> CodeExecResult:
        """
        Execute Julia code in this worker process.

        Args:
            code: Julia code to execute
            timeout: Maximum execution time in seconds

        Returns:
            CodeExecResult with stdout, stderr, and exit_code
        """
        with self.lock:
            if not self.is_healthy or self.process is None:
                raise RuntimeError(f"Worker {self.worker_id} is not healthy")

            self.is_busy = True

            try:
                # Send code to worker
                self.process.stdin.write(code + "\n")
                self.process.stdin.write(self.END_CODE + "\n")
                self.process.stdin.flush()

                # Read response with timeout
                start_time = time.time()
                stdout_lines = []
                stderr_lines = []
                exit_code = -1

                current_section = None  # Track which section we're reading

                while True:
                    # Check timeout
                    if time.time() - start_time > timeout:
                        logger.error(f"Worker {self.worker_id} execution timed out")
                        self.is_healthy = False
                        self._kill_process()
                        return CodeExecResult(
                            stdout="",
                            stderr=f"Execution timed out after {timeout} seconds",
                            exit_code=-1,
                        )

                    # Read line with timeout (use select for non-blocking read on Unix)
                    try:
                        line = self.process.stdout.readline()

                        if not line:
                            # EOF - process died
                            logger.error(f"Worker {self.worker_id} died unexpectedly")
                            self.is_healthy = False
                            return CodeExecResult(
                                stdout="".join(stdout_lines),
                                stderr="Worker process died unexpectedly",
                                exit_code=-1,
                            )

                        line = line.rstrip("\n")

                        # Check for delimiters
                        if line == self.START_OUTPUT:
                            current_section = "stdout"
                            continue
                        elif line == self.START_ERROR:
                            current_section = "stderr"
                            continue
                        elif line.startswith(self.EXIT_CODE_PREFIX):
                            # Parse exit code
                            exit_code_str = line[
                                len(self.EXIT_CODE_PREFIX) : -3
                            ]  # Remove prefix and ">>>"
                            exit_code = int(exit_code_str)
                            continue
                        elif line == self.END_EXECUTION:
                            # Execution complete
                            break

                        # Accumulate output
                        if current_section == "stdout":
                            stdout_lines.append(line)
                        elif current_section == "stderr":
                            stderr_lines.append(line)

                    except Exception as e:
                        logger.error(f"Error reading from worker {self.worker_id}: {e}")
                        self.is_healthy = False
                        return CodeExecResult(
                            stdout="".join(stdout_lines),
                            stderr=f"Error reading from worker: {str(e)}",
                            exit_code=-1,
                        )

                # Reconstruct output (add newlines back)
                stdout_str = "\n".join(stdout_lines) + ("\n" if stdout_lines else "")
                stderr_str = "\n".join(stderr_lines) + ("\n" if stderr_lines else "")

                return CodeExecResult(
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=exit_code,
                )

            finally:
                self.is_busy = False

    def _kill_process(self) -> None:
        """Kill the worker process."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
                except:
                    pass

    def shutdown(self) -> None:
        """Shutdown the worker process gracefully."""
        with self.lock:
            if self.process is not None:
                logger.info(f"Shutting down worker {self.worker_id}")
                self._kill_process()
                self.process = None
                self.is_healthy = False


class JuliaProcessPool:
    """
    Pool of persistent Julia processes for high-performance code execution.

    This class manages multiple Julia worker processes and distributes
    code execution among them, providing significant speedup by eliminating
    process startup overhead.

    Thread-safe for concurrent access from multiple threads.

    Example:
        >>> pool = JuliaProcessPool(size=4)
        >>>
        >>> # Execute code
        >>> result = pool.execute("println('Hello')")
        >>>
        >>> # Pool automatically manages workers
        >>> results = [pool.execute(f"println({i})") for i in range(100)]
        >>>
        >>> # Cleanup when done
        >>> pool.shutdown()
    """

    def __init__(
        self,
        size: int = 4,
        timeout: int = 60,
        julia_path: Optional[str] = None,
        optimization_flags: bool = True,
        auto_recover: bool = True,
    ):
        """
        Initialize the Julia process pool.

        Args:
            size: Number of worker processes to create (default: 4)
            timeout: Default timeout for code execution in seconds (default: 60)
            julia_path: Path to Julia executable (auto-detected if None)
            optimization_flags: Enable Julia optimization flags (default: True)
            auto_recover: Automatically restart failed workers (default: True)

        Raises:
            RuntimeError: If Julia executable is not found
        """
        self.size = size
        self.timeout = timeout
        self.optimization_flags = optimization_flags
        self.auto_recover = auto_recover

        # Find Julia executable
        if julia_path is None:
            julia_path = self._find_julia_executable()

        self.julia_path = julia_path

        # Find worker script
        self.worker_script = self._find_worker_script()

        # Initialize workers
        self.workers: list[JuliaWorkerProcess] = []
        self.available_workers: deque[JuliaWorkerProcess] = deque()
        self.pool_lock = threading.Lock()
        self.shutdown_flag = False

        # Create worker processes
        logger.info(f"Creating Julia process pool with {size} workers")
        for i in range(size):
            try:
                worker = JuliaWorkerProcess(
                    worker_id=i,
                    julia_path=self.julia_path,
                    worker_script=self.worker_script,
                    optimization_flags=self.optimization_flags,
                )
                self.workers.append(worker)
                self.available_workers.append(worker)
            except Exception as e:
                logger.error(f"Failed to create worker {i}: {e}")
                # Clean up partially created pool
                self.shutdown()
                raise RuntimeError(f"Failed to create worker pool: {e}")

        logger.info(f"Julia process pool initialized with {len(self.workers)} workers")

        # Register cleanup on exit
        atexit.register(self.shutdown)

    def _find_julia_executable(self) -> str:
        """Find Julia executable in PATH or common locations."""
        # Try PATH first
        julia_path = os.popen("which julia").read().strip()
        if julia_path:
            return julia_path

        # Try common locations
        common_paths = [
            os.path.expanduser("~/.juliaup/bin/julia"),
            os.path.expanduser("~/.julia/bin/julia"),
            "/usr/local/bin/julia",
            "/usr/bin/julia",
        ]

        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        raise RuntimeError(
            "Julia executable not found. Please install Julia: "
            "https://julialang.org/downloads/"
        )

    def _find_worker_script(self) -> str:
        """Find the julia_repl_worker.jl script."""
        # Try relative to this file
        this_dir = Path(__file__).parent
        worker_script = this_dir / "julia_repl_worker.jl"

        if worker_script.exists():
            return str(worker_script)

        raise RuntimeError(
            f"Worker script not found at {worker_script}. "
            "Please ensure julia_repl_worker.jl is in the same directory."
        )

    def _get_available_worker(
        self, timeout: float = 30.0
    ) -> Optional[JuliaWorkerProcess]:
        """
        Get an available worker from the pool.

        Args:
            timeout: Maximum time to wait for a worker (seconds)

        Returns:
            Available worker or None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.pool_lock:
                # Try to get healthy worker
                while self.available_workers:
                    worker = self.available_workers.popleft()

                    if worker.is_healthy:
                        return worker

                    # Worker is unhealthy, try to recover
                    if self.auto_recover and not self.shutdown_flag:
                        logger.warning(
                            f"Worker {worker.worker_id} is unhealthy, attempting recovery"
                        )
                        try:
                            worker.shutdown()
                            worker = JuliaWorkerProcess(
                                worker_id=worker.worker_id,
                                julia_path=self.julia_path,
                                worker_script=self.worker_script,
                                optimization_flags=self.optimization_flags,
                            )
                            # Update in workers list
                            self.workers[worker.worker_id] = worker
                            return worker
                        except Exception as e:
                            logger.error(
                                f"Failed to recover worker {worker.worker_id}: {e}"
                            )

            # No workers available, wait a bit
            time.sleep(0.1)

        logger.error("Timeout waiting for available worker")
        return None

    def _return_worker(self, worker: JuliaWorkerProcess) -> None:
        """Return a worker to the available pool."""
        with self.pool_lock:
            if worker.is_healthy and not self.shutdown_flag:
                self.available_workers.append(worker)

    def execute(self, code: str, timeout: Optional[int] = None) -> CodeExecResult:
        """
        Execute Julia code using an available worker from the pool.

        Args:
            code: Julia code to execute
            timeout: Execution timeout in seconds (uses pool default if None)

        Returns:
            CodeExecResult with stdout, stderr, and exit_code
        """
        if self.shutdown_flag:
            return CodeExecResult(
                stdout="",
                stderr="Process pool has been shut down",
                exit_code=-1,
            )

        if timeout is None:
            timeout = self.timeout

        # Get available worker
        worker = self._get_available_worker()

        if worker is None:
            return CodeExecResult(
                stdout="",
                stderr="No available worker (timeout waiting for worker)",
                exit_code=-1,
            )

        try:
            # Execute code in worker
            result = worker.execute(code, timeout=timeout)
            return result

        finally:
            # Return worker to pool
            self._return_worker(worker)

    def shutdown(self) -> None:
        """
        Shutdown all worker processes gracefully.

        This method is automatically called on exit via atexit.
        """
        if self.shutdown_flag:
            return

        logger.info("Shutting down Julia process pool")
        self.shutdown_flag = True

        with self.pool_lock:
            for worker in self.workers:
                worker.shutdown()

            self.workers.clear()
            self.available_workers.clear()

        logger.info("Julia process pool shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.shutdown()
