"""Hint generation for trace-aware solver RL training.

Generates concise hints from task context and rollout errors to rescue
GRPO signal on hard tasks. Follows the Self-Hinting paper approach.

Three hint modes:
- Option B: generate_hint() with verifier code + tool errors + chat_history
- Option C: generate_hint() with verifier code only (no tool errors / chat_history)
- Option D: generate_hint_from_errors() with tool errors + verifier error only
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HINT_MODEL = "anthropic/claude-sonnet-4-20250514"

HINT_SYSTEM_PROMPT = """\
You are a hint generator for tool-use tasks. Given a task description, \
its verification logic, and errors from a failed attempt, produce a single \
concise hint (2-4 sentences) that guides the solver toward the correct approach.

Rules:
- Do NOT give the full solution or exact tool call sequence.
- DO point out which tools to use, what parameters matter, or what the agent misunderstood.
- If the errors show validation failures, hint at the correct parameter format or valid options.
- If the errors show the agent used wrong tools, hint at which tools are relevant.
- If there are no errors (agent just didn't finish), hint at the general strategy.
- Keep it to a single paragraph. No bullet points, no numbered steps."""

HINT_USER_TEMPLATE = """\
## Task Prompt
{prompt}

## Verifier Logic
```python
{verifier_code}
```

## Tool Errors from Failed Attempt
{tool_errors_section}

Generate a single concise hint paragraph."""

ERROR_HINT_SYSTEM_PROMPT = """\
You are a hint generator for tool-use tasks. Given a task description \
and errors from a failed attempt (tool call errors and/or verifier failures), \
produce a single concise hint (2-4 sentences) that guides the solver toward \
the correct approach.

Rules:
- Do NOT give the full solution or exact tool call sequence.
- Synthesize both tool errors and verifier failures into actionable guidance.
- If tool errors show validation failures, hint at correct formats or valid options.
- If the verifier failed, hint at what state changes the task requires.
- Keep it to a single paragraph. No bullet points, no numbered steps."""

ERROR_HINT_USER_TEMPLATE = """\
## Task Prompt
{prompt}

## Tool Errors from Failed Attempt
{tool_errors_section}

## Verifier Failure
{verifier_error_section}

Generate a single concise hint paragraph."""

GENERIC_FALLBACK_HINT = (
    "Review the available tools carefully to understand what parameters "
    "they require, and pay attention to the exact formats and valid options "
    "for each parameter."
)


def _format_tool_errors(tool_errors: List[str]) -> str:
    """Format tool errors for inclusion in LLM prompt."""
    if not tool_errors:
        return "No tool errors recorded (agent may not have attempted relevant tools)."
    unique = list(dict.fromkeys(tool_errors))
    truncated = [e[:500] for e in unique[:10]]
    return "\n".join(f"- {e}" for e in truncated)


class HintGenerator:
    """Generates hints for solver RL training using a separate LLM.

    The hint generator is NOT the model being trained. It's a teacher model
    (e.g., Claude Sonnet) that analyzes failed rollouts and produces guidance.

    Args:
        model: litellm model identifier.
        max_tokens: Maximum tokens for hint response.
        temperature: Sampling temperature.
        api_key: Optional API key override.
    """

    def __init__(
        self,
        model: str = DEFAULT_HINT_MODEL,
        max_tokens: int = 256,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key

    async def generate_hint(
        self,
        prompt: str,
        verifier_code: str,
        tool_errors: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a hint from verifier code + task context (Options B/C).

        Option B: pass tool_errors and/or chat_history for trace-aware hints.
        Option C: omit tool_errors and chat_history for verifier-only hints.

        Args:
            prompt: The task prompt.
            verifier_code: Python verifier source code.
            tool_errors: Tool error messages from raw rollout (Option B).
            chat_history: Chat history from raw rollout (Option B, reserved).

        Returns:
            A concise hint string (single paragraph).
        """
        import litellm

        tool_errors_section = _format_tool_errors(tool_errors or [])

        user_content = HINT_USER_TEMPLATE.format(
            prompt=prompt,
            verifier_code=verifier_code[:3000],
            tool_errors_section=tool_errors_section,
        )

        messages = [
            {"role": "system", "content": HINT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return await self._call_llm(messages)

    async def generate_hint_from_errors(
        self,
        prompt: str,
        tool_errors: Optional[List[str]] = None,
        verifier_error: Optional[str] = None,
    ) -> str:
        """Generate a hint from tool errors + verifier failure (Option D).

        Lighter than generate_hint() — no verifier source code or chat history.
        The LLM synthesizes both failure signals into a coherent hint.

        Args:
            prompt: The task prompt.
            tool_errors: Tool error messages from raw rollout.
            verifier_error: Verifier execution error message.

        Returns:
            A concise hint string (single paragraph).
        """
        import litellm

        tool_errors_section = _format_tool_errors(tool_errors or [])
        verifier_error_section = verifier_error or "Verifier did not report a specific error."

        user_content = ERROR_HINT_USER_TEMPLATE.format(
            prompt=prompt,
            tool_errors_section=tool_errors_section,
            verifier_error_section=verifier_error_section,
        )

        messages = [
            {"role": "system", "content": ERROR_HINT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return await self._call_llm(messages)

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Make the LLM call and return the hint text."""
        import litellm

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = await litellm.acompletion(**kwargs)
            hint = response.choices[0].message.content.strip()
            logger.info(f"Generated hint ({len(hint)} chars)")
            return hint
        except Exception as e:
            logger.error(f"Hint generation failed: {e}")
            return GENERIC_FALLBACK_HINT


def compute_hint_reward(raw_score: float, hint_score: float) -> float:
    """Compute combined reward from raw and hinted rollout scores.

    R = (1 - raw_score) * hint_score

    - Raw succeeds (1.0): reward = 0 (task too easy, no hint needed)
    - Raw fails, hint succeeds: reward = 1.0 (hard but solvable)
    - Both fail: reward = 0 (task too hard or broken)

    Args:
        raw_score: Score from raw rollout (0.0-1.0).
        hint_score: Score from hinted rollout (0.0-1.0).

    Returns:
        Combined reward (0.0-1.0).
    """
    return (1.0 - raw_score) * hint_score
