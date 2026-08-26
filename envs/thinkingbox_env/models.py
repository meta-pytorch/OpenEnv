"""Define the public action, observation, and state models for ThinkingBox.

These Pydantic models are the only benchmark-specific values serialized across
the OpenEnv WebSocket boundary.
"""

from typing import Annotated, Any, Literal

from openenv.core import Action, ListToolsAction, Observation, State, Tool
from pydantic import BaseModel, ConfigDict, Field, model_validator, TypeAdapter


class SubmittedToolCall(BaseModel):
    """Represent one provider tool call within an assistant turn.

    Args:
        name (`str`):
            Tool name supplied by the provider.
        arguments (`dict`, *optional*):
            Parsed tool arguments.
        call_id (`str`):
            Provider call identifier.
        parse_error (`str`, *optional*):
            Provider parse error that prevents execution.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str
    parse_error: str | None = Field(
        default=None,
        description="Native provider parse error; such calls are never executed.",
    )


class ToolCallResult(BaseModel):
    """Represent one model-visible result in a parallel tool batch.

    Args:
        name (`str`):
            Tool name.
        call_id (`str`):
            Provider call identifier.
        content (`str`):
            Model-visible tool response.
        tool_error (`str`, *optional*):
            Stable public error classification.
        direct_response (`str`, *optional*):
            Native direct-response rendering.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    call_id: str
    content: str
    tool_error: Literal["tool_not_found", "invalid_args", "execution_error"] | None = (
        None
    )
    direct_response: str | None = None


class CallToolAction(Action):
    """Call one tool or one native parallel tool batch.

    Args:
        tool_name (`str`, *optional*):
            Single tool name.
        arguments (`dict`, *optional*):
            Single-call arguments.
        call_id (`str`, *optional*):
            Single provider call identifier.
        parallel_tool_calls (`list[SubmittedToolCall]`, *optional*):
            Ordered native provider batch.
    """

    type: Literal["call_tool"] = "call_tool"
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = Field(
        default=None,
        description="Provider tool-call identifier, when the model supplied one.",
    )
    parallel_tool_calls: list[SubmittedToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_single_or_parallel(self) -> "CallToolAction":
        """Require exactly one valid single-call or parallel-call representation."""
        has_single = self.tool_name is not None
        has_parallel = bool(self.parallel_tool_calls)
        if has_single == has_parallel:
            raise ValueError(
                "supply exactly one tool_name or a non-empty parallel_tool_calls batch"
            )
        if has_parallel and (self.arguments or self.call_id is not None):
            raise ValueError(
                "batch calls carry arguments and call IDs on each child call"
            )
        call_ids = [call.call_id for call in self.parallel_tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("parallel tool call IDs must be unique")
        return self


class SubmitMessageAction(Action):
    """Submit assistant-visible text or a terminal provider turn.

    Args:
        content (`str`, *optional*):
            Assistant text visible to the user and grader.
        terminal_tool_calls (`list[SubmittedToolCall]`, *optional*):
            Provider calls recorded but not executed after terminal ordering.
        tool_calls_before_content (`bool`, *optional*, defaults to `False`):
            Whether the provider emitted terminal calls before visible text.
    """

    type: Literal["submit_message"] = "submit_message"
    content: str | None = None
    terminal_tool_calls: list[SubmittedToolCall] = Field(default_factory=list)
    tool_calls_before_content: bool = False

    @model_validator(mode="after")
    def require_visible_turn(self) -> "SubmitMessageAction":
        """Require visible text or at least one terminal provider call."""
        if self.content is None and not self.terminal_tool_calls:
            raise ValueError(
                "content is required unless terminal_tool_calls are supplied"
            )
        return self


class _FinishAction(Action):
    """Trusted harness finish action; intentionally absent from public schemas."""

    type: Literal["_finish"] = "_finish"
    reason: str = "harness"


_PublicAction = Annotated[
    ListToolsAction | CallToolAction | SubmitMessageAction,
    Field(discriminator="type"),
]
_WireAction = Annotated[
    ListToolsAction | CallToolAction | SubmitMessageAction | _FinishAction,
    Field(discriminator="type"),
]
_public_action_adapter = TypeAdapter(_PublicAction)
_wire_action_adapter = TypeAdapter(_WireAction)


class ThinkingBoxAction(Action):
    """Expose the discriminated public action union used on the wire."""

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Action:  # type: ignore[override]
        """Validate a payload into its concrete ThinkingBox action.

        Args:
            obj (`object`):
                Python payload to validate.
            kwargs (`dict`, *optional*):
                Additional Pydantic validation arguments.

        Returns:
            [`~openenv.core.Action`]:
                Concrete list-tools, tool-call, message, or trusted finish action.
        """
        return _wire_action_adapter.validate_python(obj, **kwargs)

    @classmethod
    def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Return the public schema without the trusted harness finish action.

        Args:
            kwargs (`dict`, *optional*):
                Additional Pydantic schema arguments.

        Returns:
            `dict`:
                JSON schema for model-visible actions.
        """
        return _public_action_adapter.json_schema(**kwargs)


class ThinkingBoxObservation(Observation):
    """Represent the privacy-reviewed model-visible episode observation.

    Reset observations expose the task, instructions, permitted tools, and
    visible message history. Step observations add tool or simulated-user
    output, while terminal observations add binary grading status and public
    provenance without private benchmark state.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["reset", "tools", "tool", "tool_batch", "user", "terminal", "error"]
    task_uid: str | None = None
    task: str | None = None
    system_instructions: str | None = None
    bot_instructions: str | None = None
    tools: list[Tool] | None = None
    messages: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    call_id: str | None = None
    tool_result: str | None = None
    tool_error: Literal["tool_not_found", "invalid_args", "execution_error"] | None = (
        None
    )
    direct_response: str | None = None
    tool_results: list[ToolCallResult] | None = None
    user_message: str | None = None
    response: str | None = None
    finish_reason: str | None = None
    reward_type: Literal["pass", "fail", "system_error"] | None = None
    system_error: bool = False
    test_summary: dict[str, Any] | None = None
    error: str | None = None
    steps_taken: int = 0

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize the observation without absent optional fields.

        Args:
            kwargs (`dict`, *optional*):
                Additional Pydantic serialization arguments.

        Returns:
            `dict`:
                JSON-compatible observation payload.
        """
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


class ThinkingBoxState(State):
    """Represent non-sensitive OpenEnv lifecycle state.

    Args:
        task_uid (`str`, *optional*):
            Active canonical task UID.
        status (`str`, *optional*, defaults to `"idle"`):
            Current server lifecycle state.
        system_error (`bool`, *optional*, defaults to `False`):
            Whether an infrastructure failure was latched.
    """

    task_uid: str | None = None
    status: Literal["idle", "active", "finalizing", "done", "closed", "error"] = "idle"
    system_error: bool = False
