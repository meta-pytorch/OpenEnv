"""Thin Logfire wrapper for Fleet environment telemetry.

Provides structured error/event tracking for fleet task executions.
If configure_fleet_telemetry() is never called, logfire silently drops events.
"""

import logfire


def configure_fleet_telemetry(
    token=None, environment=None, service_name="openenv-fleet", **kwargs
):
    """Configure Logfire for Fleet telemetry.

    Args:
        token: Logfire API token (or set LOGFIRE_TOKEN env var).
        environment: Environment name (e.g., "production", "staging").
        service_name: Service name for Logfire (default: "openenv-fleet").
        **kwargs: Additional arguments passed to logfire.configure().
    """
    logfire.configure(
        token=token,
        service_name=service_name,
        environment=environment,
        **kwargs,
    )


def fleet_error(msg, **attrs):
    """Log a structured error event."""
    logfire.error(msg, **attrs)


def fleet_exception(msg, **attrs):
    """Log a structured error with exception info (use inside except blocks)."""
    logfire.error(msg, _exc_info=True, **attrs)


def fleet_warning(msg, **attrs):
    """Log a structured warning event."""
    logfire.warn(msg, **attrs)


def fleet_info(msg, **attrs):
    """Log a structured info event."""
    logfire.info(msg, **attrs)
