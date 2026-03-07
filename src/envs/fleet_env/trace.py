"""Fleet trace upload utilities for eval rollouts.

Provides functions to create trace jobs and upload conversation traces
to the Fleet API for viewing in the Fleet UI (including screenshots).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def create_trace_job(api_key: str, name: str) -> str:
    """Create a Fleet trace job for grouping eval traces.

    Args:
        api_key: Fleet API key.
        name: Name for the trace job (e.g. "run_name_step_100").

    Returns:
        The job_id string.
    """
    from fleet._async import AsyncFleet

    fleet = AsyncFleet(api_key=api_key)
    return await fleet.trace_job(name=name)


async def upload_trace(
    api_key: str,
    job_id: str,
    task_key: str,
    model: str,
    chat_history: List[Dict[str, Any]],
    reward: float,
    instance_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Upload a conversation trace to the Fleet API.

    Converts chat_history (OpenAI message format) to Fleet SessionIngestMessage
    format and ingests it as a trace session.

    Args:
        api_key: Fleet API key.
        job_id: Trace job ID from create_trace_job().
        task_key: Fleet task key.
        model: Model identifier (e.g. model path or name).
        chat_history: List of messages in OpenAI format (system/user/assistant).
            May contain multimodal content with image_url entries.
        reward: Episode reward (>0 = completed, else failed).
        instance_id: Optional Fleet environment instance ID.
        metadata: Optional additional metadata dict.

    Returns:
        The session_id string, or None if upload failed.
    """
    try:
        from fleet._async import AsyncFleet

        fleet = AsyncFleet(api_key=api_key)

        # Convert chat_history to ingest message format.
        # Fleet's SessionIngestMessage accepts content: Any, so OpenAI-format
        # messages (including structured content with image_url) pass through directly.
        messages = [{"role": msg["role"], "content": msg.get("content")} for msg in chat_history]

        status = "completed" if reward > 0 else "failed"

        response = await fleet._ingest(
            messages=messages,
            job_id=job_id,
            task_key=task_key,
            model=model,
            instance_id=instance_id,
            status=status,
            metadata=metadata,
        )
        return response.session_id
    except Exception as e:
        logger.warning(f"Failed to upload trace for {task_key}: {e}")
        return None
