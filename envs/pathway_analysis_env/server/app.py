# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Pathway Analysis Environment."""

import inspect
import logging
import os

# Pathway lab is meant to be used at /web; OpenEnv defaults web off unless set.
if "ENABLE_WEB_INTERFACE" not in os.environ:
    os.environ["ENABLE_WEB_INTERFACE"] = "true"

from openenv.core.env_server.http_server import create_app

from ..models import PathwayAction, PathwayObservation
from .gradio_ui import build_pathway_gradio_app
from .pathway_environment import PathwayEnvironment

_logger = logging.getLogger(__name__)

_sig = inspect.signature(create_app)
_kw: dict = {
    "env": PathwayEnvironment,
    "action_cls": PathwayAction,
    "observation_cls": PathwayObservation,
    "env_name": "pathway_analysis_env",
}
if "gradio_builder" in _sig.parameters:
    _kw["gradio_builder"] = build_pathway_gradio_app
else:
    _logger.warning(
        "openenv-core does not support gradio_builder; Pathway lab tab will be unavailable."
    )

app = create_app(**_kw)


def main():
    """Entry point for ``uv run --project . server``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
