# pyre-strict
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Deterministic testing framework for async Python code using virtual time.

Provides event loops backed by deterministic simulators for reproducible
testing of concurrent operations.
"""

from .rust_backed_event_loop import RustBackedEventLoop, RustBackedEventLoopPolicy
from .simulator_test_case import RustBackedSimulatorTestCase

__all__ = [
    "RustBackedEventLoop",
    "RustBackedEventLoopPolicy",
    "RustBackedSimulatorTestCase",
]
