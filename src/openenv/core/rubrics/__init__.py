# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rubric system for reward computation in OpenEnv environments.

See RFC 004 for design rationale: rfcs/004-rubrics.md
"""

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.trajectory import (
    TrajectoryRubric,
    ExponentialDiscountingTrajectoryRubric,
)

__all__ = [
    "Rubric",
    "TrajectoryRubric",
    "ExponentialDiscountingTrajectoryRubric",
]
