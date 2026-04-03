from typing import Any, Dict

from ..models import EmailTriageAction


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
