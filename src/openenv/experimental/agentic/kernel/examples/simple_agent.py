# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Example: Spawn a single agent, send it a message, stream the response.

This shows the simplest possible usage — just a name and system prompt.
No cluster, no image build, no port allocation.

Usage:
    python -m agentkernel.examples.simple_agent
    python -m agentkernel.examples.simple_agent --backend bwrap
    python -m agentkernel.examples.simple_agent --config agentkernel.yaml

Requires LLM_API_KEY or OPENAI_API_KEY environment variable.
Requires openclaw-agent-server binary (set OPENCLAW_RUNNER_BIN or add to PATH).

For kubernetes backend, provide a YAML config file (see agentkernel.yaml.example):
    python -m agentkernel.examples.simple_agent --config agentkernel.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import sys

from agentic.kernel import AgentKernel, SpawnRequest, TurnRequest
from agentic.kernel.plugins.openclaw import OpenClawPlugin, OpenClawSpawnInfo


async def main() -> None:
    parser = argparse.ArgumentParser(description="Simple agent example")
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
        # Spawn an agent — no image, no cluster, no port allocation needed
        result = await kernel.spawner.spawn(
            SpawnRequest(
                name="assistant",
                agent_type="openclaw",
                metadata={"role": "worker"},
                spawn_info=OpenClawSpawnInfo(
                    system_prompt="You are a helpful assistant. Be concise.",
                    tools=["bash"],
                    api_key=api_key,
                ),
            )
        )
        agent = result.agent
        print(f"Spawned agent: {agent.id} (backend={args.backend})")

        # Show agent's process environment
        info = await kernel.agent_client.get_info(agent.id)
        print(f"  pid={info['pid']}  cwd={info['cwd']}  uid={info['uid']}")
        print(f"  root fs: {', '.join(info.get('root_contents', []))}")

        # Talk to the agent (nonce is required — only the spawner can
        # communicate with the agent directly)
        request = TurnRequest(
            agent_id=agent.id,
            nonce=result.nonce,
            body=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "What is 2+2? Reply in one word."}
                    ]
                }
            ).encode(),
        )

        print("Response: ", end="", flush=True)
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
