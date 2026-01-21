# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Unity ML-Agents Environment.

The Unity environment wraps Unity ML-Agents environments (PushBlock, 3DBall,
GridWorld, etc.) providing a unified interface for reinforcement learning.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.types import Action, Observation, State


class UnityAction(Action):
    """
    Action for Unity ML-Agents environments.

    Supports both discrete and continuous action spaces. Unity environments
    may use either or both types of actions:

    - Discrete actions: Integer indices for categorical choices
      (e.g., movement direction: 0=forward, 1=backward, 2=left, 3=right)
    - Continuous actions: Float values typically in [-1, 1] range
      (e.g., joint rotations, force magnitudes)

    Example (PushBlock - discrete):
        >>> action = UnityAction(discrete_actions=[3])  # Rotate left

    Example (Walker - continuous):
        >>> action = UnityAction(continuous_actions=[0.5, -0.3, 0.0, ...])

    Attributes:
        discrete_actions: List of discrete action indices for each action branch.
            For PushBlock: [0-6] where 0=noop, 1=forward, 2=backward,
            3=rotate_left, 4=rotate_right, 5=strafe_left, 6=strafe_right
        continuous_actions: List of continuous action values, typically in [-1, 1].
        metadata: Additional action parameters.
    """

    discrete_actions: Optional[List[int]] = Field(
        default=None,
        description="Discrete action indices for each action branch",
    )
    continuous_actions: Optional[List[float]] = Field(
        default=None,
        description="Continuous action values, typically in [-1, 1] range",
    )


class UnityObservation(Observation):
    """
    Observation from Unity ML-Agents environments.

    Contains vector observations (sensor readings) and optionally visual
    observations (rendered images). Most Unity environments provide vector
    observations; visual observations are optional and must be requested.

    Attributes:
        vector_observations: Flattened array of all vector observations.
            Size and meaning depends on the specific environment.
            For PushBlock: 70 values from 14 ray-casts detecting walls/goals/blocks.
        visual_observations: Optional list of base64-encoded images (PNG format).
            Only included when include_visual=True in reset/step.
        behavior_name: Name of the Unity behavior (agent type).
        action_spec_info: Information about the action space for this environment.
        observation_spec_info: Information about the observation space.
    """

    vector_observations: List[float] = Field(
        default_factory=list,
        description="Flattened vector observations from the environment",
    )
    visual_observations: Optional[List[str]] = Field(
        default=None,
        description="Base64-encoded PNG images (when include_visual=True)",
    )
    behavior_name: str = Field(
        default="",
        description="Name of the Unity behavior/agent type",
    )
    action_spec_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the action space",
    )
    observation_spec_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the observation space",
    )


class UnityState(State):
    """
    Extended state for Unity ML-Agents environments.

    Provides additional metadata about the currently loaded environment,
    including action and observation space specifications.

    Attributes:
        episode_id: Unique identifier for the current episode.
        step_count: Number of steps taken in the current episode.
        env_id: Identifier of the currently loaded Unity environment.
        behavior_name: Name of the Unity behavior (agent type).
        action_spec: Detailed specification of the action space.
        observation_spec: Detailed specification of the observation space.
        available_envs: List of available environment identifiers.
    """

    env_id: str = Field(
        default="PushBlock",
        description="Identifier of the loaded Unity environment",
    )
    behavior_name: str = Field(
        default="",
        description="Name of the Unity behavior/agent type",
    )
    action_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specification of the action space",
    )
    observation_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specification of the observation space",
    )
    available_envs: List[str] = Field(
        default_factory=list,
        description="List of available Unity environments",
    )


# Available Unity environments from the ML-Agents registry
# These are pre-built environments that can be downloaded automatically
AVAILABLE_UNITY_ENVIRONMENTS = [
    "PushBlock",
    "3DBall",
    "3DBallHard",
    "GridWorld",
    "Basic",
    "VisualPushBlock",
    # Note: More environments may be available in newer versions of ML-Agents
]

# Action descriptions for PushBlock (most commonly used example)
PUSHBLOCK_ACTIONS = {
    0: "noop",
    1: "forward",
    2: "backward",
    3: "rotate_left",
    4: "rotate_right",
    5: "strafe_left",
    6: "strafe_right",
}
