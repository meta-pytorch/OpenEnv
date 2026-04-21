# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from opencode_env.task import OpenCodeTask


def test_coerce_from_string():
    task = OpenCodeTask.coerce("write fizzbuzz")
    assert task.instruction == "write fizzbuzz"
    assert task.setup_shell is None
    assert task.upload_files == {}


def test_coerce_from_dict():
    task = OpenCodeTask.coerce(
        {
            "instruction": "run tests",
            "setup_shell": "pip install pytest",
            "upload_files": {"/home/user/workdir/hello.py": "print('hi')"},
            "metadata": {"task_id": "hello_001"},
        }
    )
    assert task.instruction == "run tests"
    assert task.setup_shell == "pip install pytest"
    assert task.metadata["task_id"] == "hello_001"


def test_coerce_passes_through_existing_task():
    existing = OpenCodeTask(instruction="x")
    assert OpenCodeTask.coerce(existing) is existing


def test_coerce_rejects_bad_type():
    with pytest.raises(TypeError):
        OpenCodeTask.coerce(42)
