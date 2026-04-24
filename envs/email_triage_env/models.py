from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    try:
        from openenv_core.env_server.types import Action, Observation, State
    except ImportError:
        from core.env_server.types import Action, Observation, State


EmailCategory = Literal["billing", "support", "spam", "urgent", "marketing", "other"]
Difficulty = Literal["easy", "medium", "hard", "adversarial"]
TaskId = Literal["easy", "medium", "hard", "adversarial"]


class EmailTriageAction(Action):
    """Coordinator action for triaging a ticket.

    The base fields (category, priority, should_escalate) are backward-compatible
    with the Round 1 single-step environment.  The optional fields add multi-turn
    coordination metadata used in Round 2 scoring.
    """

    category: EmailCategory = Field(..., description="Predicted email category")
    priority: int = Field(..., ge=1, le=5, description="Predicted priority from 1 to 5")
    should_escalate: bool = Field(..., description="Whether the email should be escalated")

    # Round 2 optional fields — all default so old clients work unchanged
    rationale: Optional[str] = Field(
        default=None,
        description="Free-text reasoning (used for oversight quality scoring)",
    )


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
    # Round 2 multi-turn tracking
    queue_size: int = 0
    tickets_resolved: int = 0
    tickets_remaining: int = 0
    sla_breaches: int = 0
    policy_violations: int = 0
    oversight_catches: int = 0
    drift_count: int = 0
