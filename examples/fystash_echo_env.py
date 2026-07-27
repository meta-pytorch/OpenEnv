#!/usr/bin/env python3
"""Hello-world example running an OpenEnv server image on Fystash.

Boots a registry image inside a Fystash docker-template room via
``FystashProvider``, waits for ``/health``, then tears down.

Usage:
    export FYSTASH_API_KEY=key-…
    # optional: FYSTASH_API=https://api.fystash.ai FYSTASH_TEMPLATE_ID=docker
    export OPENENV_IMAGE=your-org/echo-env:latest   # registry image required (v1)
    PYTHONPATH=src uv run python examples/fystash_echo_env.py

Requires:
    A Fystash org API key (https://fystash.ai/signup → Account).
    A published OpenEnv server image (v1 does not build Dockerfiles).
"""

from __future__ import annotations

import logging
import os
import sys

from openenv.core.containers.runtime.fystash_provider import FystashProvider

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    image = os.environ.get("OPENENV_IMAGE")
    if not image:
        logger.error(
            "Set OPENENV_IMAGE to a registry tag "
            "(FystashProvider v1 does not build Dockerfiles)."
        )
        return 2
    if not os.environ.get("FYSTASH_API_KEY"):
        logger.error("Set FYSTASH_API_KEY (https://fystash.ai/signup).")
        return 2

    with FystashProvider(image=image) as provider:
        logger.info("Starting Fystash room + docker pull/run...")
        base_url = provider.start_container()
        logger.info("Preview URL: %s", base_url)
        logger.info("Waiting for /health...")
        provider.wait_for_ready(base_url, timeout_s=300)
        logger.info("Server ready at %s", base_url)
        logger.info("Stopping room...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
