# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What reaches the engine, and what must not.

Every case here produces a normal-looking response containing nothing trainable, which is the
failure this whole layer exists to prevent. None of them raise, and none of them are visible in a
rollout summary.
"""

from __future__ import annotations

import pytest

upstream = pytest.importorskip("openenv.core.harness.capture.upstream")
dialects = pytest.importorskip("openenv.core.harness.capture.dialects")
detection = pytest.importorskip("openenv.core.harness.capture.detection")

prepare_request = upstream.prepare_request


def chat_transformer():
    return dialects.TransformManager().get(detection.APIType.OPENAI_CHAT)


# --- capture parameters -----------------------------------------------------
def test_capture_parameters_are_forced_on():
    out = prepare_request({"messages": []})
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True


@pytest.mark.parametrize(
    "given,expected",
    [({}, 0), ({"top_logprobs": None}, 0), ({"top_logprobs": 5}, 5)],
)
def test_top_logprobs_is_never_left_as_none(given, expected):
    """`setdefault` keeps an explicit `None`, and vLLM then returns no logprobs at all.

    A harness sending `"top_logprobs": null` would get a perfectly normal response whose
    `logprobs.content[]` is empty, so every turn it produced would be silently untrainable.
    """
    assert prepare_request({"messages": [], **given})["top_logprobs"] == expected


def test_prior_thinking_is_renamed_for_vllm():
    """vLLM reads a prior turn's thinking from `reasoning`; the dialects emit `reasoning_content`.

    Without the rename the earlier turn renders as an empty `<think></think>`, so the prompt differs
    from what the model actually produced and prefix matching breaks for the turn after it.
    """
    out = prepare_request(
        {"messages": [{"role": "assistant", "reasoning_content": "thought"}]}
    )
    message = out["messages"][0]
    assert message["reasoning"] == "thought"
    assert "reasoning_content" not in message


# --- the internal marker ----------------------------------------------------
def test_served_model_marker_is_read_during_transform():
    """The marker has to be on the body before the transformer runs, because that is when it reads it.

    It used to be attached afterwards, by the upstream client, so every transformer saw `None`: the
    Qwen3.5 fix below never applied and the marker travelled on to the engine unused.
    """
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out = chat_transformer().transform_request(
        {**body, "_served_model": "Qwen/Qwen3.5-9B"}
    )
    assert out["chat_template_kwargs"]["enable_thinking"] is False


def test_without_the_marker_the_model_specific_fix_cannot_apply():
    """Pins the old behaviour, so the test above cannot pass for the wrong reason."""
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out = chat_transformer().transform_request(dict(body))
    assert "chat_template_kwargs" not in out


def test_the_fix_is_model_specific():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out = chat_transformer().transform_request(
        {**body, "_served_model": "meta-llama/Llama-3-8B"}
    )
    assert "enable_thinking" not in (out.get("chat_template_kwargs") or {})


def test_the_marker_never_reaches_the_engine():
    """It is an internal field. An unknown key on the wire is at best noise and at worst a 400."""
    assert "_served_model" not in chat_transformer().transform_request(
        {"model": "m", "messages": [], "_served_model": "Qwen/Qwen3.5-9B"}
    )
    assert "_served_model" not in prepare_request(
        {"messages": [], "_served_model": "leaked"}, served_model="x"
    )


# --- the caller's messages are the ones the graph stores ---------------------
def test_prepare_request_does_not_rewrite_the_callers_messages():
    """`reasoning_content` -> `reasoning` is for the engine only.

    The message dicts handed in here are the same objects the graph stores as `request_messages`.
    Renaming a field in place means the captured turn no longer records what the harness sent, so
    conversation export and any re-tokenisation downstream see the wrong shape.
    """
    message = {"role": "assistant", "content": "hi", "reasoning_content": "thinking"}
    request = {"messages": [message]}

    out = prepare_request(dict(request), served_model="m")

    assert message["reasoning_content"] == "thinking", "caller's message was mutated"
    assert "reasoning" not in message
    # The engine still gets the rename it needs.
    assert out["messages"][0]["reasoning"] == "thinking"
    assert "reasoning_content" not in out["messages"][0]


def test_messages_without_reasoning_are_passed_through_untouched():
    message = {"role": "user", "content": "go"}
    out = prepare_request({"messages": [message]})
    assert out["messages"][0] is message


# --- engine base URL --------------------------------------------------------
@pytest.mark.parametrize(
    "given",
    [
        "http://host:8000",
        "http://host:8000/",
        "http://host:8000/v1",
        "http://host:8000/v1/",
    ],
)
def test_engine_base_is_the_root_whichever_form_is_given(given):
    """Every route this layer builds is `/v1/...`, so a `/v1` base must not double up.

    `http://host:8000/v1` is what an OpenAI SDK wants and what people paste, and it used to make
    the startup probe request `/v1/v1/models` and report a healthy engine as unreachable.
    """
    assert upstream.normalise_engine_base(given) == "http://host:8000"
    assert upstream.InferenceClient(given).base_url == "http://host:8000"


# --- capture level: what a provider will actually accept --------------------
#
# Every case below was observed against a live endpoint. The `text` level exists because current
# OpenAI models answer `logprobs` with a 400, and the `tokens` level cannot be sent to them at all:
# `return_token_ids` is rejected outright, so injecting it unconditionally meant every single call
# failed rather than merely losing its token ids.
def test_tokens_level_is_the_existing_behaviour():
    out = prepare_request({"messages": []}, capture_level="tokens")
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True
    assert out["top_logprobs"] == 0


def test_logprobs_level_asks_for_logprobs_but_not_token_ids():
    out = prepare_request({"messages": []}, capture_level="logprobs")
    assert out["logprobs"] is True
    assert out["top_logprobs"] == 0
    assert "return_token_ids" not in out


def test_text_level_injects_nothing():
    out = prepare_request({"messages": [], "model": "m"}, capture_level="text")
    assert out == {"messages": [], "model": "m"}


def test_text_level_strips_prior_thinking_instead_of_renaming_it():
    """`reasoning`/`reasoning_content` is a vLLM accommodation, and a non-standard message field is
    the same 400 hazard as a non-standard top-level param."""
    out = prepare_request(
        {"messages": [{"role": "assistant", "reasoning_content": "thinking"}]},
        capture_level="text",
    )
    message = out["messages"][0]
    assert "reasoning_content" not in message
    assert "reasoning" not in message


def test_text_level_still_does_not_rewrite_the_callers_messages():
    original = {"role": "assistant", "reasoning_content": "thinking"}
    messages = [original]
    prepare_request({"messages": messages}, capture_level="text")
    assert original["reasoning_content"] == "thinking", (
        "the caller's message objects are the ones the graph stores; rewriting them in place "
        "means a captured turn no longer records what the harness sent"
    )
