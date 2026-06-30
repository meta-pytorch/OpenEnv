# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Pathway Analysis Environment."""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Pathway lab is meant to be used at /web; OpenEnv defaults web off unless set.
if "ENABLE_WEB_INTERFACE" not in os.environ:
    os.environ["ENABLE_WEB_INTERFACE"] = "true"

# Some dependencies (e.g. gseapy) import matplotlib, which tries to write a font/cache
# directory under the user's home. In sandboxed / CI contexts this can be unwritable.
if "MPLCONFIGDIR" not in os.environ:
    cache_dir = Path(__file__).resolve().parent.parent / "outputs" / ".mplcache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

from openenv.core.env_server.http_server import create_app

from ..models import PathwayAction, PathwayObservation
from .gradio_ui import build_pathway_gradio_app
from .pathway_environment import PathwayEnvironment

_logger = logging.getLogger(__name__)

# Populated when the Gradio / web UI is built (single shared env instance).
_WEB_MANAGER: Dict[str, Any] = {}


def _pathway_env_factory() -> PathwayEnvironment:
    return PathwayEnvironment()


def _gradio_builder_with_manager(web_manager, *args, **kwargs):
    _WEB_MANAGER["manager"] = web_manager
    return build_pathway_gradio_app(web_manager, *args, **kwargs)


_sig = inspect.signature(create_app)
_kw: dict = {
    "env": _pathway_env_factory,
    "action_cls": PathwayAction,
    "observation_cls": PathwayObservation,
    "env_name": "pathway_analysis_env",
}
if "gradio_builder" in _sig.parameters:
    _kw["gradio_builder"] = _gradio_builder_with_manager
else:
    _logger.warning(
        "openenv-core does not support gradio_builder; Pathway lab tab will be unavailable."
    )

app = create_app(**_kw)


def _active_pathway_env() -> Optional[PathwayEnvironment]:
    mgr = _WEB_MANAGER.get("manager")
    if mgr is None:
        return None
    env = getattr(mgr, "env", None)
    return env if isinstance(env, PathwayEnvironment) else None


@app.get(
    "/orchestrator/episode_outcome",
    tags=["Orchestrator"],
    summary="Episode score (orchestrator only)",
)
async def orchestrator_episode_outcome() -> Dict[str, Any]:
    """
    Return ``episode_outcome`` for the active web-session environment.

    Not for untrusted agents — use after ``submit_answer`` when running benchmarks
    against the local server. Returns ``{}`` if no episode has been scored yet.
    """
    env = _active_pathway_env()
    if env is None:
        return {"error": "web_interface_not_initialized"}
    return dict(env.episode_outcome or {})


@app.get(
    "/orchestrator/eval_protocol",
    tags=["Orchestrator"],
    summary="Eval protocol summary",
)
async def orchestrator_eval_protocol() -> Dict[str, Any]:
    """Describe eval-mode guarantees for the active environment instance."""
    env = _active_pathway_env()
    if env is None:
        return {"error": "web_interface_not_initialized"}
    st = env.state
    return {
        "eval_mode": st.eval_mode,
        "max_steps": st.max_steps,
        "pipeline_mode": st.pipeline_mode,
        "legacy_mode": st.legacy_mode,
        "de_run": st.de_run,
        "enrichment_run": st.enrichment_run,
        "step_count": st.step_count,
        "required_workflow": [
            "understand_experiment_design or inspect_dataset",
            "run_differential_expression",
            "run_pathway_enrichment",
            "submit_answer",
        ],
    }


def main():
    """Entry point for ``uv run --project . server``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
