# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local Python Executor.

This module provides functionality for executing Python code locally by wrapping
the smolagents LocalPythonExecutor with timeout protection.
"""

import multiprocessing
import signal
from typing import Optional

from smolagents import LocalPythonExecutor

from core.env_server.types import CodeExecResult


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Code execution timed out")


class PyExecutor:
    """
    Wrapper around smolagents LocalPythonExecutor for executing Python code.

    This class provides a simple interface to execute Python code in a subprocess
    and capture the results including stdout, stderr, and exit code.

    Args:
        additional_imports: List of additional module imports to authorize.
                          For example: ["numpy", "pandas", "matplotlib"]
                          These will be added to the base authorized imports.

    Example:
        >>> # Basic usage with default imports
        >>> executor = PyExecutor()
        >>> result = executor.run("print('Hello, World!')")
        >>> print(result.stdout)  # "Hello, World!\n"
        >>> print(result.exit_code)  # 0
        >>>
        >>> # Usage with additional imports
        >>> executor = PyExecutor(additional_imports=["numpy", "pandas"])
        >>> result = executor.run("import numpy as np\\nprint(np.array([1, 2, 3]))")
        >>> print(result.stdout)  # "[1 2 3]\n"
    """

    def __init__(self, additional_imports: list[str] | None = None):
        """
        Initialize the PyExecutor with a LocalPythonExecutor instance.

        Args:
            additional_imports: List of additional module names to authorize for import.
                              Defaults to an empty list if not provided.
        """
        if additional_imports is None:
            additional_imports = []
        self._executor = LocalPythonExecutor(
            additional_authorized_imports=additional_imports
        )
        # Initialize tools to make BASE_PYTHON_TOOLS available (including print)
        self._executor.send_tools({})

    def run(self, code: str, timeout_s: Optional[float] = None) -> CodeExecResult:
        """
        Execute Python code and return the result with optional timeout protection.

        Args:
            code: Python code string to execute
            timeout_s: Maximum execution time in seconds. If None, no timeout is enforced.
                      If the code exceeds this time, it will be terminated with a timeout error.

        Returns:
            CodeExecResult containing stdout, stderr, and exit_code

        Example:
            >>> executor = PyExecutor()
            >>> result = executor.run("x = 5 + 3\\nprint(x)")
            >>> print(result.stdout)  # "8\n"
            >>> print(result.exit_code)  # 0
            >>>
            >>> # Error handling
            >>> result = executor.run("1 / 0")
            >>> print(result.exit_code)  # 1
            >>> print(result.stderr)  # Contains error message
            >>>
            >>> # Timeout protection
            >>> result = executor.run("while True: pass", timeout_s=5.0)
            >>> print(result.exit_code)  # 1
            >>> print("timeout" in result.stderr.lower())  # True
        """
        try:
            # Set up timeout using signal (Unix/Linux only)
            old_handler = None
            if timeout_s is not None and timeout_s > 0:
                try:
                    # Set alarm signal handler for timeout
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(int(timeout_s))
                except (ValueError, AttributeError):
                    # signal.alarm is not available on Windows
                    # Fall back to no timeout on Windows
                    pass

            try:
                # Execute the code using LocalPythonExecutor
                # LocalPythonExecutor returns a CodeOutput object with output, logs, is_final_answer
                exec_result = self._executor(code)

                # Extract the logs (which contain print outputs) as stdout
                # The output field contains the return value of the code
                stdout = exec_result.logs
                stderr = ""
                exit_code = 0  # Success

                return CodeExecResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                )
            finally:
                # Cancel the alarm and restore old handler
                if timeout_s is not None and timeout_s > 0:
                    try:
                        signal.alarm(0)
                        if old_handler is not None:
                            signal.signal(signal.SIGALRM, old_handler)
                    except (ValueError, AttributeError):
                        pass

        except TimeoutError as e:
            # Code execution exceeded timeout
            return CodeExecResult(
                stdout="",
                stderr=f"Code execution timed out after {timeout_s} seconds. "
                       f"Possible infinite loop or extremely long computation.",
                exit_code=1,  # Non-zero indicates error
            )

        except Exception as e:
            # LocalPythonExecutor raises InterpreterError for various issues
            # (syntax errors, forbidden operations, runtime errors, etc.)
            return CodeExecResult(
                stdout="",
                stderr=str(e),
                exit_code=1,  # Non-zero indicates error
            )
