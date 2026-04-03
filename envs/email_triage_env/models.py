from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from core.env_server.types import Action, Observation, State


EmailCategory = Literal["billing", "support", "spam", "urgent", "marketing", "other"]
Difficulty = Literal["easy", "medium", "hard"]
TaskId = Literal["easy", "medium", "hard"]


class EmailTriageAction(Action):
    category: EmailCategory = Field(..., description="Predicted email category")
    priority: int = Field(..., ge=1, le=5, description="Predicted priority from 1 to 5")
    should_escalate: bool = Field(..., description="Whether the email should be escalated")


class EmailTriageObservation(Observation):
    email_id: str
    subject: str
    body_snippet: str
    sender: str
    sender_domain: str
    is_internal: bool
    task_id: TaskId
    info: Optional[Dict[str, Any]] = None


class EmailTriageState(State):
    total_reward: float = 0.0
    difficulty: Difficulty = "medium"
    current_task: TaskId = "medium"
