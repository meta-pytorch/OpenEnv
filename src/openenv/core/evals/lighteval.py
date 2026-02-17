# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LightEval harness integration for OpenEnv.

Requires the ``lighteval`` package: ``pip install lighteval[accelerate]>=0.13.0``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.evals.base import EvalHarness


class LightEvalHarness(EvalHarness):
    """Evaluation harness wrapping HuggingFace's LightEval Pipeline API.

    All ``lighteval`` imports are deferred to :meth:`run` so this class is
    importable without lighteval installed.  An ``ImportError`` with a clear
    message is raised at call time if the dependency is missing.

    Args:
        output_dir: Directory for evaluation outputs. Defaults to None.
        save_details: Whether to save per-sample details. Defaults to False.

    ``eval_parameters`` keys accepted by :meth:`run`:

    +--------------------------+----------+-----------------+-----------------------------------+
    | Key                      | Type     | Default         | Purpose                           |
    +==========================+==========+=================+===================================+
    | ``model_name``           | str      | *required*      | HuggingFace model identifier      |
    | ``tasks``                | str      | ``dataset`` arg | Task string, e.g. ``"gsm8k|5"``  |
    | ``model_backend``        | str      | ``transformers``| transformers/vllm/tgi/openai      |
    | ``launcher_type``        | str      | ``NONE``        | Parallelism launcher               |
    | ``max_samples``          | int|None | None            | Limit samples per task            |
    | ``num_fewshot_seeds``    | int      | 1               | Fewshot random seeds              |
    | ``custom_tasks_directory``| str|None| None            | Path to custom task definitions   |
    | ``model_args``           | dict     | ``{}``          | Kwargs for model config           |
    +--------------------------+----------+-----------------+-----------------------------------+
    """

    def __init__(
        self,
        *,
        output_dir: Optional[str] = None,
        save_details: bool = False,
    ):
        self.output_dir = output_dir
        self.save_details = save_details

    def run(
        self,
        harness_version: str,
        library_versions: Dict[str, str],
        dataset: str,
        eval_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a LightEval evaluation pipeline.

        Args:
            harness_version: Version of lighteval being used.
            library_versions: Versions of supporting libraries.
            dataset: Default dataset/task string (used when ``tasks`` is not
                specified in *eval_parameters*).
            eval_parameters: See class docstring for accepted keys.

        Returns:
            Dictionary mapping metric names to scores.

        Raises:
            ImportError: If ``lighteval`` is not installed.
            ValueError: If ``model_name`` is missing from *eval_parameters*,
                or if ``model_backend`` is not a supported value.
        """
        try:
            from lighteval.pipeline import (
                EnvConfig,
                Pipeline,
                PipelineParameters,
            )
        except ImportError:
            raise ImportError(
                "lighteval is required for LightEvalHarness. "
                "Install it with: pip install 'lighteval[accelerate]>=0.13.0'"
            )

        # Extract parameters
        model_name = eval_parameters.get("model_name")
        if model_name is None:
            raise ValueError(
                "eval_parameters must include 'model_name' "
                "(a HuggingFace model identifier)."
            )

        tasks = eval_parameters.get("tasks", dataset)
        model_backend = eval_parameters.get("model_backend", "transformers")
        launcher_type = eval_parameters.get("launcher_type", "NONE")
        max_samples = eval_parameters.get("max_samples")
        num_fewshot_seeds = eval_parameters.get("num_fewshot_seeds", 1)
        custom_tasks_directory = eval_parameters.get("custom_tasks_directory")
        model_args = eval_parameters.get("model_args", {})

        # Build model config
        model_config = self._build_model_config(model_backend, model_name, model_args)

        # Build pipeline parameters
        pipeline_params = PipelineParameters(
            launcher_type=launcher_type,
            env_config=EnvConfig(cache_dir=self.output_dir),
            custom_tasks_directory=custom_tasks_directory,
            max_samples=max_samples,
            num_fewshot_seeds=num_fewshot_seeds,
        )

        # Create and run pipeline
        pipeline = Pipeline(
            tasks=tasks,
            pipeline_parameters=pipeline_params,
            model_config=model_config,
        )
        pipeline.evaluate()
        pipeline.save_and_push_results()

        # Extract scores from results
        scores = pipeline.get_results()

        return scores

    def _build_model_config(
        self,
        model_backend: str,
        model_name: str,
        model_args: Dict[str, Any],
    ) -> Any:
        """Build a LightEval model config for the given backend.

        Args:
            model_backend: One of ``"transformers"``, ``"vllm"``, ``"tgi"``,
                ``"openai"``, ``"sglang"``.
            model_name: HuggingFace model identifier or API model name.
            model_args: Additional keyword arguments for the model config.

        Returns:
            A LightEval model config instance.

        Raises:
            ValueError: If *model_backend* is not supported.
        """
        from lighteval.models.model_config import (
            InferenceEndpointModelConfig,
            OpenAIModelConfig,
            SGLangModelConfig,
            TransformersModelConfig,
            VLLMModelConfig,
        )

        backend_map = {
            "transformers": TransformersModelConfig,
            "vllm": VLLMModelConfig,
            "tgi": InferenceEndpointModelConfig,
            "openai": OpenAIModelConfig,
            "sglang": SGLangModelConfig,
        }

        config_cls = backend_map.get(model_backend)
        if config_cls is None:
            raise ValueError(
                f"Unsupported model_backend: {model_backend!r}. "
                f"Supported backends: {sorted(backend_map)}."
            )

        return config_cls(model=model_name, **model_args)
