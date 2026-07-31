# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The interception proxy must capture per-token ids from a vLLM logprobs response.

vLLM with ``--return-tokens-as-token-ids`` encodes the id in the ``token`` field as
``"token_id:<int>"`` (no separate ``token_id`` key), which the capture must parse so
GRPO trains on real token ids."""

import os
import sys

# Make ``envs/`` importable when running from the repository root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_ENVS_DIR = os.path.join(_REPO_ROOT, "envs")
if _ENVS_DIR not in sys.path:
    sys.path.insert(0, _ENVS_DIR)

from opencode_env.sandbox.interception import _build_turn_record


def _response(content):
    return {"choices": [{"finish_reason": "stop", "logprobs": {"content": content}}]}


def test_captures_ids_from_vllm_token_id_prefix():
    # vLLM --return-tokens-as-token-ids: id lives in the ``token`` field as "token_id:N".
    content = [
        {"token": "token_id:15339", "logprob": -0.1},
        {"token": "token_id:1917", "logprob": -0.2},
    ]
    rec = _build_turn_record(
        turn_idx=0, request_body={}, response_json=_response(content), latency_s=0.0
    )
    assert rec.completion_token_ids == [15339, 1917]
    assert rec.per_token_logps == [-0.1, -0.2]


def test_plain_openai_has_logprobs_but_no_ids():
    # A plain OpenAI response (no ids) still yields logprobs, just no token ids.
    content = [{"token": "hi", "logprob": -0.3}]
    rec = _build_turn_record(
        turn_idx=0, request_body={}, response_json=_response(content), latency_s=0.0
    )
    assert rec.completion_token_ids == []
    assert rec.per_token_logps == [-0.3]
