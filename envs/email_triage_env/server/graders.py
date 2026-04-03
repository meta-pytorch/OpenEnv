from typing import Any, Dict

try:
    from envs.email_triage_env.models import EmailTriageAction, TaskId
except ImportError:
    from models import EmailTriageAction, TaskId


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def category_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """
    Returns 1.0 if predicted category matches true_category,
    0.0 otherwise.
    """
    return 1.0 if action.category == email["true_category"] else 0.0


def priority_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """
    Returns:
      1.0 if same bucket (low/med/high)
      0.5 if off by 1 bucket
      0.0 otherwise
    """

    def bucket(p: int) -> int:
        if p <= 2:
            return 0
        if p == 3:
            return 1
        return 2

    true_b = bucket(email["true_priority"])
    act_b = bucket(action.priority)
    if true_b == act_b:
        return 1.0
    if abs(true_b - act_b) == 1:
        return 0.5
    return 0.0


def escalation_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """
    Returns 1.0 if escalation decision matches,
    lower for harmful mismatches (spam escalated, urgent ignored).
    """
    if action.should_escalate == email["needs_escalation"]:
        return 1.0

    if email["true_category"] == "spam" and action.should_escalate:
        return 0.0
    if email["true_category"] == "urgent" and not action.should_escalate:
        return 0.0

    return 0.5


def easy_task_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Easy task: category classification only."""
    return category_grader(action, email)


def medium_task_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Medium task: category plus priority bucket quality."""
    score = 0.7 * category_grader(action, email) + 0.3 * priority_grader(action, email)
    return _clamp_01(score)


def hard_task_grader(action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Hard task: full triage quality with safety-sensitive escalation."""
    score = (
        0.5 * category_grader(action, email)
        + 0.2 * priority_grader(action, email)
        + 0.3 * escalation_grader(action, email)
    )

    # Apply a stronger penalty for clearly harmful mistakes.
    if email["true_category"] == "spam" and action.should_escalate:
        score -= 0.3
    if email["true_category"] == "urgent" and not action.should_escalate:
        score -= 0.3

    return _clamp_01(score)


def task_grader(task_id: TaskId, action: EmailTriageAction, email: Dict[str, Any]) -> float:
    """Dispatch to one of the three deterministic task graders."""
    if task_id == "easy":
        return easy_task_grader(action, email)
    if task_id == "medium":
        return medium_task_grader(action, email)
    return hard_task_grader(action, email)
