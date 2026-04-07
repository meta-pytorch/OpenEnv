import json
import os
import random
import uuid
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server import Environment
except ImportError:
    try:
        from openenv_core.env_server import Environment
    except ImportError:
        from core.env_server import Environment

try:
    from envs.email_triage_env.models import (
        Difficulty,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        TaskId,
    )
except ImportError:
    from models import (
        Difficulty,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        TaskId,
    )

try:
    from envs.email_triage_env.server.graders import (
        category_grader,
        escalation_grader,
        priority_grader,
        task_grader,
    )
except ImportError:
    from server.graders import (
        category_grader,
        escalation_grader,
        priority_grader,
        task_grader,
    )


TASK_CONFIG: Dict[TaskId, Dict[str, Any]] = {
    "easy": {
        "difficulty": "easy",
        "description": "Classify the email category correctly.",
        "reward_weights": {"category": 1.0, "priority": 0.1, "escalation": 0.0},
    },
    "medium": {
        "difficulty": "medium",
        "description": "Classify category and set the right priority bucket.",
        "reward_weights": {"category": 0.8, "priority": 0.3, "escalation": 0.1},
    },
    "hard": {
        "difficulty": "hard",
        "description": "Full triage: category, priority, and safe escalation behavior.",
        "reward_weights": {"category": 0.6, "priority": 0.3, "escalation": 0.3},
    },
}


class EmailTriageEnvironment(
    Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, difficulty: Difficulty = "medium") -> None:
        super().__init__()
        self._difficulty: Difficulty = difficulty if difficulty in TASK_CONFIG else "medium"
        self._task_id: TaskId = self._difficulty
        self._current_email: Dict[str, Any] = {}
        self._emails: List[Dict[str, Any]] = self._load_email_dataset()
        self._state = EmailTriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            difficulty=self._difficulty,
            current_task=self._task_id,
        )

    @staticmethod
    def task_metadata() -> Dict[TaskId, Dict[str, Any]]:
        return TASK_CONFIG

    def _load_email_dataset(self) -> List[Dict[str, Any]]:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "email_triage_dataset.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def _sample_email(self) -> Dict[str, Any]:
        candidates = [e for e in self._emails if e.get("difficulty") == self._difficulty]
        if not candidates:
            candidates = self._emails
        return random.choice(candidates)

    def _resolve_task(self, **kwargs: Any) -> TaskId:
        requested_task = kwargs.get("task_id")
        requested_difficulty = kwargs.get("difficulty")

        if requested_task in TASK_CONFIG:
            return requested_task
        if requested_difficulty in TASK_CONFIG:
            return requested_difficulty
        return self._task_id

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        if seed is not None:
            random.seed(seed)

        self._task_id = self._resolve_task(**kwargs)
        self._difficulty = TASK_CONFIG[self._task_id]["difficulty"]

        self._state = EmailTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            difficulty=self._difficulty,
            current_task=self._task_id,
        )
        self._current_email = self._sample_email()

        return self._make_observation(
            reward=0.0,
            done=False,
            info={
                "reason": "reset",
                "task_id": self._task_id,
                "task_description": TASK_CONFIG[self._task_id]["description"],
            },
        )

    def step(
        self,
        action: EmailTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        self._state.step_count += 1

        if not self._current_email:
            self._current_email = self._sample_email()

        reward = self._compute_reward(action, self._current_email)
        self._state.total_reward += reward

        task_score = task_grader(self._task_id, action, self._current_email)
        info = {
            "task_id": self._task_id,
            "task_description": TASK_CONFIG[self._task_id]["description"],
            "task_score": task_score,
            "true_category": self._current_email["true_category"],
            "true_priority": self._current_email["true_priority"],
            "true_needs_escalation": self._current_email["needs_escalation"],
            "category_score": category_grader(action, self._current_email),
            "priority_score": priority_grader(action, self._current_email),
            "escalation_score": escalation_grader(action, self._current_email),
        }

        return self._make_observation(reward=reward, done=True, info=info)

    def _make_observation(
        self, reward: float, done: bool, info: Dict[str, Any]
    ) -> EmailTriageObservation:
        body = self._current_email["body"]
        snippet = body[:280]

        return EmailTriageObservation(
            email_id=self._current_email["id"],
            subject=self._current_email["subject"],
            body_snippet=snippet,
            sender=self._current_email["sender"],
            sender_domain=self._current_email["sender_domain"],
            is_internal=self._current_email["is_internal"],
            task_id=self._task_id,
            reward=reward,
            done=done,
            metadata={
                "difficulty": self._current_email["difficulty"],
                "task_id": self._task_id,
            },
            info=info,
        )

    @property
    def state(self) -> EmailTriageState:
        return self._state

    def _compute_reward(self, action: EmailTriageAction, email: Dict[str, Any]) -> float:
        cat_score = category_grader(action, email)
        pri_score = priority_grader(action, email)
        esc_score = escalation_grader(action, email)

        weights = TASK_CONFIG[self._task_id]["reward_weights"]
        reward = 0.0
        reward += weights["category"] * cat_score
        reward += weights["priority"] * pri_score
        reward += weights["escalation"] * esc_score

        # Add task-level deterministic score to shape progress.
        reward += 0.25 * task_grader(self._task_id, action, email)

        if email["true_category"] == "spam" and action.should_escalate:
            reward -= 0.5
        if email["true_category"] == "urgent" and not action.should_escalate:
            reward -= 0.5

        return reward
