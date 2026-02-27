"""Thin Logfire wrapper for Fleet environment telemetry.

Provides structured error/event tracking for fleet task executions.
If configure_fleet_telemetry() is never called, logfire silently drops events.

All events include a consistent base schema:
- env_key: Environment key (e.g., "github", "amazon")
- env_version: Environment version (e.g., "v0.0.12")
- task_key: Task identifier
- modality: "tool_use" or "computer_use"
"""

import logfire
from contextvars import ContextVar
from typing import Optional

# Session context - set once per rollout/task execution
_session_context: ContextVar[dict] = ContextVar("fleet_session_context", default={})


def configure_fleet_telemetry(
    token: Optional[str] = None,
    environment: str = "training_rollouts",
    service_name: str = "openenv-fleet",
    **kwargs,
):
    """Configure Logfire for Fleet telemetry.

    Args:
        token: Logfire API token (or set LOGFIRE_TOKEN env var).
        environment: Environment name (default: "training_rollouts").
        service_name: Service name for Logfire (default: "openenv-fleet").
        **kwargs: Additional arguments passed to logfire.configure().
    """
    logfire.configure(
        token=token,
        service_name=service_name,
        environment=environment,
        **kwargs,
    )


def set_task_context(
    *,
    env_key: Optional[str] = None,
    env_version: Optional[str] = None,
    task_key: Optional[str] = None,
    modality: Optional[str] = None,
):
    """Set the task context for all subsequent telemetry events.

    Call this at the start of each rollout/task execution.
    """
    ctx = {}
    if env_key:
        ctx["env_key"] = env_key
    if env_version:
        ctx["env_version"] = env_version
    if task_key:
        ctx["task_key"] = task_key
    if modality:
        ctx["modality"] = modality
    _session_context.set(ctx)


def clear_task_context():
    """Clear the task context."""
    _session_context.set({})


def _with_context(**attrs) -> dict:
    """Merge session context with event-specific attributes."""
    ctx = _session_context.get().copy()
    ctx.update(attrs)
    return ctx


def fleet_info(msg: str, **attrs):
    """Log a structured info event."""
    logfire.info(msg, **_with_context(**attrs))


def fleet_warning(msg: str, **attrs):
    """Log a structured warning event."""
    logfire.warn(msg, **_with_context(**attrs))


def fleet_error(msg: str, **attrs):
    """Log a structured error event."""
    logfire.error(msg, **_with_context(**attrs))


def fleet_exception(msg: str, **attrs):
    """Log a structured error with exception info (use inside except blocks)."""
    logfire.error(msg, _exc_info=True, **_with_context(**attrs))
