# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loop-owning sessions backed by a deployed `harbor_env` server.

    factory = HarborSessionFactory(server_url="http://harbor:8000", split="...", llm_url="http://vllm:8000")
    session = factory.create(prompt)      # prompt carries the task's instruction
    session.wait_for_completion()         # the server runs the agent; this blocks
    trace = session.fetch_proxy_trace()   # per-turn records for a trainer
    reward = session.verify(transcript)   # the task's own verifier, forwarded

This is the shape TRL's `HarnessRolloutWorker` consumes in loop-owning mode, and it is deliberately
the same shape `opencode_env.harness` offers — so the training script is the stock one and nothing has
to be added to TRL. The difference is where the work happens: opencode_env runs the agent locally,
while this hands a task to a server that owns the sandbox, the agent and the capture proxy. The
trainer needs no sandbox credentials, no harness installed, and no capture plumbing of its own.

WHY THE ENGINE IS AN ARGUMENT HERE. `run_rollout` takes `llm_url` per call, so a trainer points the
server at the vLLM it is currently syncing weights into. The server probes that engine and the tier
follows from what it can return: token ids plus processed logprobs give a trainable rollout, anything
less gives an eval one. That is what lets a training run and an eval run share one server.

ONE HONEST LIMITATION. `fetch_proxy_trace` returns TRL's `TraceEntry` shape, which has no field for
the prompt's token ids, so TRL re-renders each prompt with `apply_chat_template`. The completions are
exact — they come straight off the engine — but the prompts are not guaranteed to be. Whether that
matters is measurable rather than arguable: `measure_prompt_skew` below compares the re-render against
the engine's own `prompt_token_ids` for the model and harness you are actually using. If it is 1.0 the
path is lossless; if it is not, `_chain_to_sequences` forks the conversation at the first divergence,
so it degrades visibly rather than silently.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    Message,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from openenv.harbor.client import HarborEnv
from openenv.harbor.models import HarborRolloutResult

logger = logging.getLogger(__name__)


def instruction_id(instruction: str) -> str:
    """Stable id for a task, from its instruction text.

    TRL's loop-owning path forwards only `prompt` to the factory — extra dataset columns never arrive,
    and `seed` is the group counter rather than a dataset index. So the task has to be recoverable from
    the prompt itself. Hashing the instruction keeps the prompt a real prompt (readable in logged
    completions, and what the agent is actually asked to do) instead of smuggling an index through it.
    """
    return hashlib.sha1(instruction.strip().encode()).hexdigest()


def _openai_tool_calls(flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`{name, arguments}` -> OpenAI's `{id, type, function: {name, arguments}}`.

    `HarborTurn.tool_calls` is deliberately flattened: a reward function checking which tool ran should
    not have to walk a wire envelope. But TRL reads `message["tool_calls"]` verbatim, and both
    `has_tool_call` and `apply_chat_template` expect the nested form. Handing over the flat shape makes
    `has_tool_call` false for every turn, so `train_turn_fn=has_tool_call` — the documented default for
    a coding agent — discards the entire rollout while the run still looks healthy. Templates that
    iterate `call.function.name` would raise instead.

    `arguments` is left exactly as captured. It is a JSON *string* on the wire, and TRL's
    `_decode_tool_call_arguments` parses it before rendering, so parsing it here would hand the
    template a dict it does not expect.
    """
    out: list[dict[str, Any]] = []
    for index, call in enumerate(flat or []):
        if not isinstance(call, dict):
            continue
        # Already nested (a future capture change, or another dialect): pass it through untouched.
        if call.get("function"):
            out.append(call)
            continue
        name = call.get("name")
        if not name:
            continue
        out.append(
            {
                # An id is required by the schema and is what pairs a call with its tool result. The
                # capture does not keep the harness's own id, so a positional one is minted: within a
                # single assistant message that is enough to keep the pairing unambiguous.
                "id": str(call.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": call.get("arguments", ""),
                },
            }
        )
    return out


def to_trace_entries(result: HarborRolloutResult) -> list[dict[str, Any]]:
    """`HarborRolloutResult` -> TRL `TraceEntry` records, one per trainable turn.

    Auxiliary calls and discarded retries are already excluded by the server, so a caller needs no
    `agent_turn_fn`: the capture layer can tell an aux call from an agent turn structurally, which a
    flat trace cannot.

    `request_messages` is what makes this possible at all. Without it the token fields say what was
    produced but not what produced them, and no `TraceEntry` can be built.
    """
    entries: list[dict[str, Any]] = []
    for turn in result.turns or []:
        if turn.discarded or not turn.trainable or not turn.completion_token_ids:
            continue
        entries.append(
            {
                "request": {
                    "messages": list(turn.request_messages),
                    "tools": turn.request_tools,
                },
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": turn.text,
                                "tool_calls": _openai_tool_calls(turn.tool_calls)
                                or None,
                            },
                            "finish_reason": turn.finish_reason,
                        }
                    ]
                },
                "completion_token_ids": list(turn.completion_token_ids),
                "per_token_logps": list(turn.per_token_logps),
            }
        )
    return entries


def _decoded_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool-call `arguments` as a mapping rather than a JSON string, for chat-template rendering.

    The wire format keeps `arguments` as a string, which is what the OpenAI schema specifies. XML-style
    templates — Qwen3.5's among them — iterate it, and iterating a string raises
    `Can only get item pairs from a mapping`, so a render that should be a measurement becomes a crash.

    TRL does the same thing before rendering (`_decode_tool_call_arguments`), so skipping it here would
    also mean measuring something other than what TRL actually feeds the template.
    """
    import json as _json

    out: list[dict[str, Any]] = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            out.append(message)
            continue
        decoded = []
        for call in calls:
            function = call.get("function") or {}
            arguments = function.get("arguments", call.get("arguments"))
            if not isinstance(arguments, str):
                decoded.append(call)
                continue
            try:
                parsed = _json.loads(arguments or "{}")
            except ValueError:
                # Not JSON. Pass it through: a template that cannot render it should fail loudly
                # rather than have this function invent a shape.
                decoded.append(call)
                continue
            new_call = dict(call)
            if call.get("function"):
                new_call["function"] = {**function, "arguments": parsed}
            else:
                new_call["arguments"] = parsed
            decoded.append(new_call)
        out.append({**message, "tool_calls": decoded})
    return out


def measure_prompt_skew(
    result: HarborRolloutResult, tokenizer, **template_kwargs
) -> dict[str, Any]:
    """How exactly a local re-render reproduces the engine's own prompt, per turn.

    The number nobody has by default. TRL re-renders prompts because `TraceEntry` carries no prompt
    ids; this says what that costs for a given model and harness instead of assuming it is free or
    assuming it is fatal. `exact_match_frac == 1.0` means the loop-owning path is lossless here.

    Returns:
        `dict` with keys:
            - `turns` (`int`): turns compared.
            - `exact_match_frac` (`float`): fraction whose re-render matched token for token.
            - `worst_common_prefix` (`int`): shortest agreeing prefix seen, in tokens.
            - `length_deltas` (`list[int]`): re-rendered length minus the engine's, per turn.
    """
    compared = 0
    exact = 0
    worst_prefix = -1
    deltas: list[int] = []
    for turn in result.turns or []:
        if turn.discarded or not turn.prompt_token_ids or not turn.request_messages:
            continue
        rendered = tokenizer.apply_chat_template(
            _decoded_arguments(turn.request_messages),
            tools=turn.request_tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **template_kwargs,
        )
        engine = list(turn.prompt_token_ids)
        compared += 1
        deltas.append(len(rendered) - len(engine))
        if rendered == engine:
            exact += 1
        prefix = 0
        for a, b in zip(rendered, engine):
            if a != b:
                break
            prefix += 1
        worst_prefix = prefix if worst_prefix < 0 else min(worst_prefix, prefix)
    return {
        "turns": compared,
        "exact_match_frac": (exact / compared) if compared else 0.0,
        "worst_common_prefix": max(worst_prefix, 0),
        "length_deltas": deltas,
    }


class HarborSession(ResourceSession):
    """One Harbor rollout, run by the server, read back here.

    Created idle rather than already-running: `wait_for_completion` is what starts it. The server call
    is a single blocking request that boots the sandbox, runs the agent to completion and grades the
    workspace, so there is nothing useful to do between `create` and `wait`.
    """

    def __init__(
        self,
        *,
        env: HarborEnv,
        split: str,
        task_index: int,
        instruction: str,
        harness: str,
        sandbox: str,
        llm_url: str,
        model: str,
        reward_key: str = "",
        api_key: str = "",
        auth_header: str = "",
        agent_timeout_sec: float = 0.0,
    ) -> None:
        self._env = env
        self._split = split
        self._task_index = task_index
        self._instruction = instruction
        self._harness = harness
        self._sandbox = sandbox
        self._llm_url = llm_url
        self._model = model
        self._reward_key = reward_key
        self._api_key = api_key
        self._auth_header = auth_header
        self._agent_timeout_sec = agent_timeout_sec
        self.result: HarborRolloutResult | None = None

    # --- ResourceSession -----------------------------------------------------

    def initial_messages(self) -> list[Message]:
        return [{"role": "user", "content": self._instruction}]

    def list_tools(self) -> list[Tool]:
        # The agent owns its own tool loop inside the sandbox; none are exposed to the harness.
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(
            error=(
                "HarborSession does not expose external tool calls; the agent owns its own loop "
                "inside the server's sandbox."
            )
        )

    def verify(
        self, transcript: list[Message], final_state: Any | None = None
    ) -> VerifyResult:
        """The task's own verifier score, as the server reported it.

        Never recomputed here, and never defaulted to 0: a rollout that did not run has no grade, and
        scoring that as zero teaches a policy that a crashed rollout is as good as a wrong answer.
        """
        if self.result is None:
            return VerifyResult(env_reward=None, done=True)
        return VerifyResult(env_reward=self.result.reward, done=True)

    def close(self) -> None:
        # The server owns the sandbox and tears it down with the trial, so there is nothing to release
        # here. Kept because the contract requires it, and a silent no-op is better than a subclass
        # that forgets it exists.
        return None

    # --- loop-owning extensions ---------------------------------------------

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Run the rollout and block until the server is done. Returns a process-style exit code.

        `0` means the server reported a usable rollout. Non-zero means it did not, and the reason is on
        `self.result.error` — returned rather than raised, because a rollout failing is an outcome the
        trainer has to score around, while an exception here would take down the rollout loop and with
        it every training rank waiting on the next batch.
        """
        try:
            self.result = self._env.run_rollout(
                split=self._split,
                task_index=self._task_index,
                harness=self._harness,
                sandbox=self._sandbox,
                reward_key=self._reward_key,
                llm_url=self._llm_url,
                model=self._model,
                api_key=self._api_key,
                auth_header=self._auth_header,
                # `is not None`, not `or`: 0 is a documented value meaning "defer to the task
                # file", and `or` silently replaces it with the factory default. `OpenCodeSession`
                # takes the same care for the same reason.
                agent_timeout_sec=(
                    timeout_s if timeout_s is not None else self._agent_timeout_sec
                ),
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.warning(
                "harbor rollout call failed for %s[%d]: %s: %s",
                self._split,
                self._task_index,
                type(exc).__name__,
                exc,
            )
            self.result = None
            return 1
        if not self.result.ok:
            logger.warning(
                "harbor rollout %s[%d] not ok: %s",
                self._split,
                self._task_index,
                self.result.error,
            )
        return 0 if self.result.ok else 1

    def fetch_proxy_trace(self) -> list[dict[str, Any]]:
        """Per-turn captured records, in TRL's `TraceEntry` shape.

        Empty for an eval rollout, and that is the point: an eval endpoint yields a reward and a
        readable trace but no token fields, so there is nothing to train on and this says so by being
        empty rather than by handing back rows of zeros.
        """
        if self.result is None:
            return []
        if self.result.rollout_type != "train":
            logger.warning(
                "rollout is %s (capture_level=%s), so it carries no trainable turns",
                self.result.rollout_type,
                self.result.capture_level,
            )
            return []
        return to_trace_entries(self.result)


class HarborSessionFactory(ResourceSessionFactory[HarborSession]):
    """Turns a prompt into a Harbor rollout on a deployed server.

    Args:
        server_url (`str`):
            Root of a running `harbor_env` server, e.g. `http://localhost:8000`.
        split (`str`, *optional*):
            Which dataset the server should draw tasks from. Defaults to the server's first.
        llm_url (`str`, *optional*):
            Engine for these rollouts — for training, the same vLLM the trainer syncs weights into,
            which is what makes the rollouts on-policy. Omit to use the server's default engine.
        harness (`str`, *optional*, defaults to `"opencode"`):
            Which agent runs in the sandbox.
        sandbox (`str`, *optional*, defaults to `"e2b"`):
            Harbor sandbox backend.
        agent_timeout_sec (`float`, *optional*, defaults to `600.0`):
            Ceiling on one rollout. Worth setting: a rollout holds a generation slot for the length of
            the call, and the task file's own timeout covers the agent run but not sandbox setup.

    Examples:

    ```python
    factory = HarborSessionFactory(
        server_url="http://localhost:8000",
        split="AdithyaSK/data_agent_rl_environment_train",
        llm_url="http://localhost:8001",
        model="Qwen/Qwen3.5-2B",
    )
    ```
    """

    def __init__(
        self,
        server_url: str,
        *,
        split: str = "",
        llm_url: str = "",
        model: str = "",
        harness: str = "opencode",
        sandbox: str = "e2b",
        reward_key: str = "",
        api_key: str = "",
        auth_header: str = "",
        agent_timeout_sec: float = 600.0,
        num_tasks: int | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.harness = harness
        self.sandbox = sandbox
        self.llm_url = llm_url
        self.model = model
        self.reward_key = reward_key
        self.api_key = api_key
        self.auth_header = auth_header
        self.agent_timeout_sec = agent_timeout_sec
        self._num_tasks = num_tasks
        self._env: HarborEnv | None = None
        self._split = split
        self._by_instruction: dict[str, int] = {}
        self._tasks: list[dict[str, Any]] = []

    def __getstate__(self) -> dict[str, Any]:
        """Drop the live client when this factory is pickled.

        TRL spawns its rollout loop in a separate process and pickles the factory into it. Building
        the dataset in the parent calls `prompt_rows()` -> `tasks()` -> `_client()`, which binds an
        httpx client and a websocket — so without this the pickle either fails outright or the child
        inherits a connection owned by another process and every rollout dies on it.

        The task list and the instruction map are kept: they are plain data, they cost a round trip to
        rebuild, and the child needs the same mapping the parent's dataset was built from.
        """
        state = dict(self.__dict__)
        state["_env"] = None
        return state

    # Built lazily so it is created in whichever process actually uses it, and dropped on pickling by
    # `__getstate__` above.
    def _client(self) -> HarborEnv:
        if self._env is None:
            self._env = HarborEnv(base_url=self.server_url)
            if not self._split:
                splits = self._env.splits()
                if not splits:
                    raise RuntimeError(f"{self.server_url} serves no splits")
                self._split = splits[0]["name"]
        return self._env

    def tasks(self) -> list[dict[str, Any]]:
        """The tasks this factory can run, fetched once from the server."""
        if not self._tasks:
            env = self._client()
            total = env.num_tasks(self._split)
            stop = min(total, self._num_tasks) if self._num_tasks else total
            self._tasks = [
                t.model_dump() if hasattr(t, "model_dump") else dict(t)
                for t in env.get_task_range(self._split, 0, stop)
            ]
            self._by_instruction = {}
            collisions: dict[str, int] = {}
            for i, task in enumerate(self._tasks):
                key = instruction_id(task.get("instruction") or "")
                index = int(task.get("index", i))
                if key in self._by_instruction:
                    collisions[key] = collisions.get(key, 1) + 1
                    continue
                self._by_instruction[key] = index
            if collisions:
                # Last-write-wins here would be invisible and wrong: two tasks with identical
                # instructions produce two dataset rows that both resolve to one index, so a group
                # trains on a task it was not given while every log line looks normal. Keeping the
                # first and saying how many were shadowed at least makes it findable.
                logger.warning(
                    "%d task(s) share an instruction with an earlier task and are unreachable "
                    "through prompt lookup; the first occurrence wins",
                    sum(collisions.values()) - len(collisions),
                )
        return self._tasks

    def prompt_rows(self) -> list[dict[str, Any]]:
        """Dataset rows for a trainer: the instruction as the prompt, plus columns worth logging.

        All `num_generations` of a group share a row, so they all get the same task and the group
        baseline is well formed without any seed plumbing.
        """
        return [
            {
                "prompt": [{"role": "user", "content": t.get("instruction") or ""}],
                "task_name": t.get("task_name") or "",
                "task_index": int(t.get("index", i)),
            }
            for i, t in enumerate(self.tasks())
        ]

    def create(
        self,
        task: Any,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> HarborSession:
        instruction = _instruction_of(task)
        self.tasks()  # ensures the instruction -> index map exists
        index = self._by_instruction.get(instruction_id(instruction))
        if index is None:
            raise KeyError(
                "this prompt does not match any task on the server. Build the dataset from "
                "`prompt_rows()` so the instruction the trainer sends is the one the server has."
            )
        return HarborSession(
            env=self._client(),
            split=self._split,
            task_index=index,
            instruction=instruction,
            harness=self.harness,
            sandbox=self.sandbox,
            llm_url=self.llm_url,
            model=self.model,
            reward_key=self.reward_key,
            api_key=self.api_key,
            auth_header=self.auth_header,
            agent_timeout_sec=self.agent_timeout_sec,
        )


def _instruction_of(task: Any) -> str:
    """The instruction text out of whatever the worker passed as `task`."""
    if isinstance(task, list) and task:
        last = task[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")
    if isinstance(task, dict):
        return str(task.get("instruction") or task.get("content") or "")
    return str(task or "")
