# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reasoning Gym Environment - A pure MCP environment for reasoning tasks.

This environment exposes all functionality through MCP tools:
- `get_question()`: Returns the current question and task type
- `submit_answer(answer)`: Submits answer, returns score and correct answer
- `get_task_info()`: Returns available tasks and configuration

Example:
    >>> from reasoning_gym_env import ReasoningGymEnv
    >>>
    >>> with ReasoningGymEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     question = env.call_tool("get_question")
    ...     print(question["question"])
    ...     result = env.call_tool("submit_answer", answer="8")
    ...     print(f"Score: {result['score']}")
"""

from .client import ReasoningGymEnv

__all__ = ["ReasoningGymEnv"]
