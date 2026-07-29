# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the LaTeX OCR Task API."""

import pytest
from latex_ocr_env.server import latex_ocr_environment as environment_module
from latex_ocr_env.server.latex_ocr_environment import LatexOCREnvironment


def test_stream_list_tasks_is_explicitly_unsupported(monkeypatch):
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: 150)
    env = LatexOCREnvironment(mode="stream")

    with pytest.raises(NotImplementedError, match="stream mode"):
        env.list_tasks("train")


def test_get_task_range_supports_python_slice_bounds(monkeypatch):
    monkeypatch.setattr(environment_module, "_load_split", lambda *_args: [None] * 5)
    env = LatexOCREnvironment(mode="materialize")

    assert [task["index"] for task in env.get_task_range("train", -2)] == [3, 4]
    assert [task["index"] for task in env.get_task_range("train", None, -1)] == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("num_tasks", ("validation",)),
        ("list_tasks", ("validation",)),
        ("get_task", ("validation", 0)),
        ("get_task_range", ("validation", 0, 1)),
    ],
)
def test_task_api_rejects_unconfigured_splits(monkeypatch, method_name, args):
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: -1)
    env = LatexOCREnvironment(mode="stream")

    with pytest.raises(ValueError, match="unknown split"):
        getattr(env, method_name)(*args)


def test_stream_task_range_rejects_oversized_requests(monkeypatch):
    monkeypatch.setattr(environment_module, "_split_total", lambda *_args: 20)
    monkeypatch.setattr(environment_module, "_STREAM_RANGE_CAP", 10)
    env = LatexOCREnvironment(mode="stream")

    with pytest.raises(ValueError, match="at most 10"):
        env.get_task_range("train", 0, 11)
