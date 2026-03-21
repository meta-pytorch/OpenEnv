# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Pathway Analysis Environment."""

from openenv.core.env_server import create_app

from ..models import PathwayAction, PathwayObservation
from .pathway_environment import PathwayEnvironment

app = create_app(
    PathwayEnvironment,
    PathwayAction,
    PathwayObservation,
    env_name="pathway_analysis_env",
)


def main():
    """Entry point for ``uv run --project . server``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
