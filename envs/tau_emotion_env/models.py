# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State for the τ-emotion-bench environment.

The environment is MCP-only: use `ListToolsAction` and `CallToolAction` from
`openenv.core.env_server.mcp_types` to interact with it.
"""

from typing import Any

from openenv.core.env_server import State


class TauEmotionState(State):
    """
    Episode state.

    Attributes:
        domain: τ-emotion-bench domain of this environment.
        split: Task split tasks are drawn from.
        task_id: The task of the current episode.
        done: Whether the conversation has ended.
        termination_reason: Why it ended, e.g. `user_stop`, `user_unsatisfied`,
            `agent_stop` or `max_steps`. Empty while the conversation runs.
        reward: τ-emotion-bench's reward, set when the conversation ends. `-1.0`
            when the user gave up on the agent.
        reward_info: τ-emotion-bench's evaluation breakdown.
    """

    domain: str = ""
    split: str = ""
    task_id: str = ""
    done: bool = False
    termination_reason: str = ""
    reward: float = 0.0
    reward_info: dict[str, Any] = {}
