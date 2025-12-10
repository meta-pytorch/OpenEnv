"""
envs/julia_env/julia_transforms.py
--------------------------------
Safety and quality transforms for Julia code.
"""

import re
from core.env_server.base_transforms import CompositeTransform
from core.env_server.interfaces import Transform
from ..models import JuliaObservation


# -------------------------
# Safety Transform
# -------------------------
class JuliaSafetyTransform(Transform):
    """Detects dangerous Julia operations and penalizes them with a negative reward."""

    def __init__(self, penalty: float = -3.0):
        self.penalty = penalty
        self.dangerous_patterns = [
            r"run\(",
            r"read\(",
            r"write\(",
            r"unsafe_",
            r"ccall\(",
            r"Base\.exit",
            r"Base\.kill",
            r"rm\(",      # file deletion
            r"download\(" # downloading
        ]

    def __call__(self, observation):
        # Only act on JuliaObservation objects
        if not isinstance(observation, JuliaObservation):
            return observation

        # Extract last executed code from metadata
        code = observation.metadata.get("last_code", "") if observation.metadata else ""

        for pattern in self.dangerous_patterns:
            if re.search(pattern, code):
                # Apply penalty and record violation
                observation.reward = (observation.reward or 0.0) + self.penalty
                observation.metadata = observation.metadata or {}
                observation.metadata["safety_violation"] = pattern
                return observation

        # Safe code gets neutral reward
        observation.reward = observation.reward or 0.0
        return observation


# -------------------------
# Quality Transform
# -------------------------
class JuliaQualityTransform(Transform):
    """Evaluates and rewards Julia code quality."""

    def __init__(self, concise_bonus=1, max_length_threshold=120):
        self.concise_bonus = concise_bonus
        self.max_length_threshold = max_length_threshold

    def __call__(self, observation):
        # Only act on JuliaObservation objects
        if not isinstance(observation, JuliaObservation):
            return observation

        code = observation.metadata.get("last_code", "") if observation.metadata else ""
        reward = observation.reward or 0.0

        # Reward concise code
        if len(code.strip()) <= self.max_length_threshold:
            reward += self.concise_bonus
        else:
            reward -= 0.1  # slight penalty for verbosity

        observation.reward = reward
        return observation


# -------------------------
# Composite Transform
# -------------------------
def create_safe_julia_transform():
    """Combines safety and quality transforms into one pipeline."""
    return CompositeTransform([JuliaSafetyTransform(), JuliaQualityTransform()])
