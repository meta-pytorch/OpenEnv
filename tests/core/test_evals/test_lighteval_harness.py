# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for LightEvalHarness with mocked LightEval dependencies."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from openenv.core.evals import EvalConfig, EvalResult
from openenv.core.evals.lighteval import LightEvalHarness


# ---------------------------------------------------------------------------
# Helpers to build mock lighteval modules
# ---------------------------------------------------------------------------


def _make_mock_lighteval_modules():
    """Build a dict of mock modules that simulate lighteval's structure."""
    # Top-level
    lighteval_mod = ModuleType("lighteval")

    # lighteval.pipeline
    pipeline_mod = ModuleType("lighteval.pipeline")
    pipeline_mod.Pipeline = MagicMock(name="Pipeline")
    pipeline_mod.PipelineParameters = MagicMock(name="PipelineParameters")
    pipeline_mod.EnvConfig = MagicMock(name="EnvConfig")

    # lighteval.models and lighteval.models.model_config
    models_mod = ModuleType("lighteval.models")
    model_config_mod = ModuleType("lighteval.models.model_config")
    model_config_mod.TransformersModelConfig = MagicMock(name="TransformersModelConfig")
    model_config_mod.VLLMModelConfig = MagicMock(name="VLLMModelConfig")
    model_config_mod.InferenceEndpointModelConfig = MagicMock(
        name="InferenceEndpointModelConfig"
    )
    model_config_mod.OpenAIModelConfig = MagicMock(name="OpenAIModelConfig")
    model_config_mod.SGLangModelConfig = MagicMock(name="SGLangModelConfig")

    return {
        "lighteval": lighteval_mod,
        "lighteval.pipeline": pipeline_mod,
        "lighteval.models": models_mod,
        "lighteval.models.model_config": model_config_mod,
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestLightEvalHarnessConstruction:
    """Test instantiation and default values."""

    def test_default_construction(self):
        harness = LightEvalHarness()
        assert harness.output_dir is None
        assert harness.save_details is False

    def test_custom_construction(self):
        harness = LightEvalHarness(output_dir="/tmp/evals", save_details=True)
        assert harness.output_dir == "/tmp/evals"
        assert harness.save_details is True

    def test_name_property(self):
        harness = LightEvalHarness()
        assert harness.name == "LightEvalHarness"

    def test_is_eval_harness_subclass(self):
        from openenv.core.evals.base import EvalHarness

        assert issubclass(LightEvalHarness, EvalHarness)


class TestLightEvalHarnessImportGuard:
    """Test that run() raises a clear ImportError when lighteval is missing."""

    def test_import_error_message(self):
        harness = LightEvalHarness()
        with patch.dict(sys.modules, {"lighteval": None, "lighteval.pipeline": None}):
            with pytest.raises(ImportError, match="lighteval is required"):
                harness.run(
                    harness_version="0.13.0",
                    library_versions={},
                    dataset="gsm8k",
                    eval_parameters={"model_name": "gpt2"},
                )


class TestLightEvalHarnessRun:
    """Test the run() method with mocked lighteval."""

    def _run_harness(self, eval_parameters, dataset="gsm8k", **init_kwargs):
        """Helper to run the harness with mocked lighteval modules."""
        mock_modules = _make_mock_lighteval_modules()
        mock_pipeline_cls = mock_modules["lighteval.pipeline"].Pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.get_results.return_value = {"acc": 0.85}
        mock_pipeline_cls.return_value = mock_pipeline_instance

        harness = LightEvalHarness(**init_kwargs)
        with patch.dict(sys.modules, mock_modules):
            scores = harness.run(
                harness_version="0.13.0",
                library_versions={"transformers": "4.36.0"},
                dataset=dataset,
                eval_parameters=eval_parameters,
            )

        return scores, mock_pipeline_cls, mock_pipeline_instance, mock_modules

    def test_basic_run_returns_scores(self):
        scores, _, _, _ = self._run_harness({"model_name": "gpt2"})
        assert scores == {"acc": 0.85}

    def test_pipeline_created_with_correct_tasks(self):
        _, mock_cls, _, _ = self._run_harness(
            {"model_name": "gpt2"},
            dataset="hellaswag",
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["tasks"] == "hellaswag"

    def test_tasks_parameter_overrides_dataset(self):
        _, mock_cls, _, _ = self._run_harness(
            {"model_name": "gpt2", "tasks": "gsm8k|5"},
            dataset="hellaswag",
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["tasks"] == "gsm8k|5"

    def test_evaluate_called_but_save_skipped_by_default(self):
        _, _, mock_instance, _ = self._run_harness({"model_name": "gpt2"})
        mock_instance.evaluate.assert_called_once()
        mock_instance.save_and_push_results.assert_not_called()
        mock_instance.get_results.assert_called_once()

    def test_save_called_when_save_details_enabled(self):
        _, _, mock_instance, _ = self._run_harness(
            {"model_name": "gpt2"}, save_details=True
        )
        mock_instance.evaluate.assert_called_once()
        mock_instance.save_and_push_results.assert_called_once()
        mock_instance.get_results.assert_called_once()

    def test_missing_model_name_raises_value_error(self):
        harness = LightEvalHarness()
        mock_modules = _make_mock_lighteval_modules()
        with patch.dict(sys.modules, mock_modules):
            with pytest.raises(ValueError, match="model_name"):
                harness.run(
                    harness_version="0.13.0",
                    library_versions={},
                    dataset="gsm8k",
                    eval_parameters={},
                )

    def test_default_model_backend_is_transformers(self):
        _, _, _, mock_modules = self._run_harness({"model_name": "gpt2"})
        model_config_mod = mock_modules["lighteval.models.model_config"]
        model_config_mod.TransformersModelConfig.assert_called_once_with(model="gpt2")

    def test_output_dir_passed_to_env_config(self):
        _, mock_cls, _, mock_modules = self._run_harness(
            {"model_name": "gpt2"},
            output_dir="/tmp/evals",
        )
        env_config_cls = mock_modules["lighteval.pipeline"].EnvConfig
        env_config_cls.assert_called_once_with(cache_dir="/tmp/evals")

    def test_pipeline_parameters_defaults(self):
        _, mock_cls, _, mock_modules = self._run_harness({"model_name": "gpt2"})
        pp_cls = mock_modules["lighteval.pipeline"].PipelineParameters
        call_kwargs = pp_cls.call_args[1]
        assert call_kwargs["launcher_type"] == "NONE"
        assert call_kwargs["max_samples"] is None
        assert call_kwargs["num_fewshot_seeds"] == 1
        assert call_kwargs["custom_tasks_directory"] is None

    def test_custom_pipeline_parameters(self):
        _, _, _, mock_modules = self._run_harness(
            {
                "model_name": "gpt2",
                "launcher_type": "ACCELERATE",
                "max_samples": 50,
                "num_fewshot_seeds": 3,
                "custom_tasks_directory": "/path/to/tasks",
            }
        )
        pp_cls = mock_modules["lighteval.pipeline"].PipelineParameters
        call_kwargs = pp_cls.call_args[1]
        assert call_kwargs["launcher_type"] == "ACCELERATE"
        assert call_kwargs["max_samples"] == 50
        assert call_kwargs["num_fewshot_seeds"] == 3
        assert call_kwargs["custom_tasks_directory"] == "/path/to/tasks"


class TestLightEvalHarnessModelBackends:
    """Test model backend mapping."""

    def _run_with_backend(self, backend, model_args=None):
        mock_modules = _make_mock_lighteval_modules()
        mock_pipeline_cls = mock_modules["lighteval.pipeline"].Pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.get_results.return_value = {}
        mock_pipeline_cls.return_value = mock_pipeline_instance

        harness = LightEvalHarness()
        params = {"model_name": "gpt2", "model_backend": backend}
        if model_args:
            params["model_args"] = model_args

        with patch.dict(sys.modules, mock_modules):
            harness.run(
                harness_version="0.13.0",
                library_versions={},
                dataset="test",
                eval_parameters=params,
            )
        return mock_modules["lighteval.models.model_config"]

    def test_transformers_backend(self):
        mc = self._run_with_backend("transformers")
        mc.TransformersModelConfig.assert_called_once_with(model="gpt2")

    def test_vllm_backend(self):
        mc = self._run_with_backend("vllm")
        mc.VLLMModelConfig.assert_called_once_with(model="gpt2")

    def test_tgi_backend(self):
        mc = self._run_with_backend("tgi")
        mc.InferenceEndpointModelConfig.assert_called_once_with(model="gpt2")

    def test_openai_backend(self):
        mc = self._run_with_backend("openai")
        mc.OpenAIModelConfig.assert_called_once_with(model="gpt2")

    def test_sglang_backend(self):
        mc = self._run_with_backend("sglang")
        mc.SGLangModelConfig.assert_called_once_with(model="gpt2")

    def test_unsupported_backend_raises_value_error(self):
        harness = LightEvalHarness()
        mock_modules = _make_mock_lighteval_modules()
        with patch.dict(sys.modules, mock_modules):
            with pytest.raises(ValueError, match="Unsupported model_backend"):
                harness.run(
                    harness_version="0.13.0",
                    library_versions={},
                    dataset="test",
                    eval_parameters={
                        "model_name": "gpt2",
                        "model_backend": "unknown",
                    },
                )

    def test_model_args_passed_through(self):
        mc = self._run_with_backend(
            "transformers",
            model_args={"dtype": "float16", "batch_size": 8},
        )
        mc.TransformersModelConfig.assert_called_once_with(
            model="gpt2", dtype="float16", batch_size=8
        )


class TestLightEvalHarnessIntegration:
    """Test run_from_config produces correct EvalResult."""

    def test_run_from_config_returns_eval_result(self):
        mock_modules = _make_mock_lighteval_modules()
        mock_pipeline_cls = mock_modules["lighteval.pipeline"].Pipeline
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.get_results.return_value = {
            "acc": 0.85,
            "acc_stderr": 0.02,
        }
        mock_pipeline_cls.return_value = mock_pipeline_instance

        harness = LightEvalHarness()
        config = EvalConfig(
            harness_name="LightEvalHarness",
            harness_version="0.13.0",
            library_versions={"transformers": "4.36.0"},
            dataset="gsm8k",
            eval_parameters={"model_name": "gpt2"},
        )

        with patch.dict(sys.modules, mock_modules):
            result = harness.run_from_config(config)

        assert isinstance(result, EvalResult)
        assert result.config is config
        assert result.scores["acc"] == 0.85
        assert result.scores["acc_stderr"] == 0.02
