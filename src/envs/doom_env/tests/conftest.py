# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pytest configuration and shared fixtures for Doom environment tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_server: marks tests that require a running server"
    )
    config.addinivalue_line(
        "markers", "requires_vizdoom: marks tests that require ViZDoom installed"
    )
    config.addinivalue_line(
        "markers", "requires_docker: marks tests that require Docker"
    )
    config.addinivalue_line(
        "markers", "requires_display: marks tests that require display/X11"
    )


@pytest.fixture
def sample_observation():
    """Create a sample DoomObservation for testing."""
    from ..models import DoomObservation

    return DoomObservation(
        screen_buffer=[128] * 300,  # 10x10x3
        screen_shape=[10, 10, 3],
        game_variables=[100.0, 50.0],
        available_actions=[0, 1, 2, 3],
        episode_finished=False,
        done=False,
        reward=0.0,
        metadata={"episode": 1}
    )


@pytest.fixture
def sample_action():
    """Create a sample DoomAction for testing."""
    from ..models import DoomAction

    return DoomAction(action_id=0)


@pytest.fixture
def sample_screen_buffer_160x120():
    """Create a sample screen buffer for 160x120 resolution."""
    # RGB24: 120 height x 160 width x 3 channels
    return [128] * (120 * 160 * 3)


@pytest.fixture
def sample_screen_buffer_640x480():
    """Create a sample screen buffer for 640x480 resolution."""
    # RGB24: 480 height x 640 width x 3 channels
    return [255] * (480 * 640 * 3)
