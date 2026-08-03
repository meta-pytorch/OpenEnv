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
