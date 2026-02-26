# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Example: Spawn a team of agents with a shared image and have them communicate.

This shows the image path — when you need to bundle code or skills into agents.
The architect gets just a system prompt (no image), while workers share an image
built with custom code bundles.

Usage:
    python -m agentkernel.examples.team_scenario
    python -m agentkernel.examples.team_scenario --backend bwrap
    python -m agentkernel.examples.team_scenario --config agentkernel.yaml

Requires LLM_API_KEY or OPENAI_API_KEY environment variable.
Requires openclaw-agent-server binary (set OPENCLAW_RUNNER_BIN or add to PATH).

For kubernetes backend, provide a YAML config file (see agentkernel.yaml.example):
    python -m agentkernel.examples.team_scenario --config agentkernel.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from agentic.kernel import (
    AgentKernel,
    CreateTeamRequest,
    SourceBundle,
    SpawnRequest,
    TurnRequest,
)
from agentic.kernel.plugins.openclaw import OpenClawPlugin, OpenClawSpawnInfo


async def main() -> None:
    parser = argparse.ArgumentParser(description="Team scenario example")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "bwrap", "kubernetes"],
        help="Spawner backend (default: local)",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config file (required for kubernetes backend)",
    )
    args = parser.parse_args()

    log_level = os.environ.get("AGENTKERNEL_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s - %(message)s",
    )

    plugins = [OpenClawPlugin()]

    if args.config:
        kernel = AgentKernel.from_config(args.config, plugins=plugins)
    elif args.backend == "kubernetes":
        parser.error("--config is required when using --backend kubernetes")
    else:
        kernel = AgentKernel(backend=args.backend, plugins=plugins)

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: LLM_API_KEY or OPENAI_API_KEY environment variable must be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Reserve capacity for the team
        await kernel.spawner.create_team(
            CreateTeamRequest(
                team_id="project-team",
                resources={"cpu": 4},
            )
        )

        # Create a code bundle for workers (e.g., shared utilities)
        with tempfile.TemporaryDirectory() as tmpdir:
            helpers_dir = Path(tmpdir) / "worker_helpers"
            helpers_dir.mkdir()
            (helpers_dir / "tools.py").write_text(
                "def estimate_effort(task: str) -> str:\n"
                "    return f'Estimated effort for: {task}'\n"
            )
            # Add a requirements.txt to demonstrate dependency installation
            # in the OCI image (the Dockerfile will pip install these)
            (helpers_dir / "requirements.txt").write_text("requests>=2.28\n")
            bundle_uri = kernel.blob_store.upload_dir(helpers_dir)

        # Package a worker image with the code bundle
        worker_job = await kernel.packaging.create_agent_image(
            name="worker",
            bundles=[SourceBundle(uri=bundle_uri, labels={"name": "worker_helpers"})],
        )
        if worker_job.status != "succeeded" or worker_job.image is None:
            raise RuntimeError(
                f"Failed to build worker image: {worker_job.error or 'unknown error'}"
            )
        worker_image_id = worker_job.image.id
        print(f"Built worker image: {worker_image_id} ({worker_job.image.path})")

        # Spawn all agents in parallel — architect + workers
        spawn_requests = [
            SpawnRequest(
                name="architect",
                team_id="project-team",
                agent_type="openclaw",
                metadata={"role": "coordinator"},
                spawn_info=OpenClawSpawnInfo(
                    system_prompt=(
                        "You are a software architect. You coordinate a team "
                        "of engineers. Break down tasks and assign them to workers."
                    ),
                    tools=["bash"],
                    api_key=api_key,
                ),
            ),
        ]
        for name in ["backend-eng", "frontend-eng"]:
            spawn_requests.append(
                SpawnRequest(
                    name=name,
                    team_id="project-team",
                    agent_type="openclaw",
                    metadata={"role": "worker"},
                    image_id=worker_image_id,
                    spawn_info=OpenClawSpawnInfo(
                        system_prompt=(
                            f"You are a {name}. Implement what is asked of you."
                        ),
                        tools=["bash"],
                        api_key=api_key,
                    ),
                ),
            )

        all_results = await asyncio.gather(
            *(kernel.spawner.spawn(req) for req in spawn_requests)
        )
        architect_result = all_results[0]
        worker_results = list(all_results[1:])
        print(f"Spawned architect: {architect_result.agent.id}")
        for wr in worker_results:
            print(f"Spawned worker: {wr.agent.name} ({wr.agent.id})")

        # Architect assigns work
        for wr in worker_results:
            worker = wr.agent
            task = f"Build the {worker.name} component."
            request = TurnRequest(
                agent_id=worker.id,
                nonce=wr.nonce,
                body=json.dumps(
                    {"messages": [{"role": "user", "content": task}]}
                ).encode(),
            )

            print(f"\nArchitect -> {worker.name}: {task}")
            print(f"{worker.name} responds: ", end="", flush=True)
            async for chunk in kernel.agent_client.turn(request):
                if chunk.body:
                    print(chunk.body, end="", flush=True)
                if chunk.error:
                    print(f"\nError: {chunk.error}", flush=True)
            print()

    finally:
        await kernel.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
