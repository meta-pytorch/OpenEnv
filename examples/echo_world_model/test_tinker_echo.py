"""Unit tests for the Tinker-backed ECHO example (no API key required)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from backends.tinker_echo_demo import build_echo_data, mean_env_ce
from trajectory import ACTION, CONTEXT, ENV_OUTPUT, WARNING, Segment, Trajectory


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


@dataclass
class _ModelInput:
    tokens: list[int]

    @classmethod
    def from_ints(cls, tokens):
        return cls(tokens)


@dataclass
class _Datum:
    model_input: _ModelInput
    loss_fn_inputs: dict


_TINKER_TYPES = SimpleNamespace(Datum=_Datum, ModelInput=_ModelInput)


def test_build_echo_data_applies_shifted_env_only_mask():
    trajectory = Trajectory(
        segments=[
            Segment(CONTEXT, "C"),
            Segment(ACTION, "AB"),
            Segment(WARNING, "W"),
            Segment(ENV_OUTPUT, "XY"),
        ],
        reward=1.0,
    )

    [datum] = build_echo_data(_TINKER_TYPES, _CharTokenizer(), [trajectory])

    assert datum.model_input.tokens == [ord(char) for char in "CABWX"]
    assert datum.loss_fn_inputs["target_tokens"] == [ord(char) for char in "ABWXY"]
    assert datum.loss_fn_inputs["weights"] == [0.0, 0.0, 0.0, 0.5, 0.5]


def test_build_echo_data_normalizes_weights_across_batch():
    trajectories = [
        Trajectory([Segment(ENV_OUTPUT, "abc")], reward=0.0),
        Trajectory([Segment(CONTEXT, "x"), Segment(ENV_OUTPUT, "de")], reward=0.0),
    ]

    data = build_echo_data(_TINKER_TYPES, _CharTokenizer(), trajectories)

    assert sum(sum(datum.loss_fn_inputs["weights"]) for datum in data) == pytest.approx(
        1.0
    )


def test_build_echo_data_rejects_trajectory_without_env_targets():
    trajectory = Trajectory([Segment(ACTION, "answer")], reward=1.0)

    with pytest.raises(ValueError, match="env_output"):
        build_echo_data(_TINKER_TYPES, _CharTokenizer(), [trajectory])


def test_build_echo_data_accepts_real_tinker_types():
    tinker = pytest.importorskip("tinker")
    trajectory = Trajectory(
        [Segment(CONTEXT, "prompt"), Segment(ENV_OUTPUT, "result")], reward=0.0
    )

    [datum] = build_echo_data(tinker.types, _CharTokenizer(), [trajectory])

    assert isinstance(datum, tinker.types.Datum)
    assert datum.model_input.to_ints() == [ord(char) for char in "promptresul"]
    assert sum(datum.loss_fn_inputs["weights"].tolist()) == pytest.approx(1.0)


class _Tensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


def test_mean_env_ce_uses_only_weighted_tokens():
    data = [
        _Datum(
            model_input=_ModelInput([1, 2, 3]),
            loss_fn_inputs={"weights": _Tensor([0.0, 0.25, 0.75])},
        )
    ]
    result = SimpleNamespace(
        loss_fn_outputs=[{"logprobs": _Tensor([-99.0, -2.0, -4.0])}]
    )

    assert mean_env_ce(result, data) == pytest.approx(3.5)
