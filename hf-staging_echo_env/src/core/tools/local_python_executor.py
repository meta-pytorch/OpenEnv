# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local Python Executor.

This module provides functionality for executing Python code locally by wrapping
the smolagents LocalPythonExecutor.
"""

from smolagents import LocalPythonExecutor

from core.env_server.types import CodeExecResult


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

    def run(self, code: str) -> CodeExecResult:
        """
        Execute Python code and return the result.

        Args:
            code: Python code string to execute

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
        """
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

        except Exception as e:
            # LocalPythonExecutor raises InterpreterError for various issues
            # (syntax errors, forbidden operations, runtime errors, etc.)
            return CodeExecResult(
                stdout="",
                stderr=str(e),
                exit_code=1,  # Non-zero indicates error
            )
