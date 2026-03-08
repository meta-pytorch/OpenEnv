"""Fleet trace upload utilities for eval rollouts.

Provides functions to create trace jobs and upload conversation traces
to the Fleet API for viewing in the Fleet UI (including screenshots).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _convert_image_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI image_url block to Fleet ingest image format.

    Fleet ingest API expects: {"type": "image", "mime_type": "image/png", "data": "<base64>"}
    It then uploads base64 to S3 and replaces with URL for the UI to render.
    """
    url = block.get("image_url", {}).get("url", "")
    if url.startswith("data:"):
        # data:image/png;base64,ABC... -> extract mime_type and base64 data
        header, base64_data = url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        return {"type": "image", "mime_type": mime_type, "data": base64_data}
    else:
        # HTTPS URL - pass as text since ingest API expects base64 for images
        return {"type": "text", "text": url}


def _convert_content(content: Any) -> Any:
    """Convert OpenAI-format content blocks to Anthropic format for Fleet UI."""
    if not isinstance(content, list):
        return content
    converted = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            converted.append(_convert_image_block(block))
        else:
            converted.append(block)
    return converted


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
        # Fleet UI expects Anthropic content block format, so we convert
        # OpenAI image_url blocks to Anthropic image blocks.
        messages = [
            {"role": msg["role"], "content": _convert_content(msg.get("content"))}
            for msg in chat_history
        ]

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
