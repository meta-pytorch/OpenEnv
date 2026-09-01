"""Run one ThinkingBox case with the agent configured in a native config.

This trusted example hydrates the selected public task locally while all
private episode state and grading remain inside the OpenEnv server.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from eval_testlist import (
    configured_agent_session_factory,
    load_thinkingbox_config,
    run_configured_agent,
)
from thinkingbox.common.hydrator import get_dataset_case_by_name
from thinkingbox_env import ThinkingBoxEnv
from thinkingbox_env.server.data_loader import resolve_data_bundle


async def main() -> None:
    """Parse command-line settings and run one configured benchmark episode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("test_uid")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--env-config",
        help="Server-visible config path; defaults to --config.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--env-dataset",
        help="Server-visible data path; defaults to --dataset.",
    )
    parser.add_argument("--agent", default="think")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_thinkingbox_config(config_path)
    bundle = resolve_data_bundle(args.dataset)
    test_case = get_dataset_case_by_name(
        args.test_uid,
        base_dir=bundle.dataset_dir,
        agent=args.agent,
    )

    async with ThinkingBoxEnv(base_url=args.base_url) as env:
        result = await env.reset(
            args.test_uid,
            dataset=args.env_dataset or args.dataset,
            agent=args.agent,
            config=args.env_config or str(config_path),
        )
        print("task:", result.observation.task)
        print("tools:", [tool.name for tool in result.observation.tools or []])

        if not result.done:
            result = await run_configured_agent(
                env,
                result,
                test_case,
                configured_agent_session_factory(config),
            )

        print("reward:", result.reward)
        print("finish_reason:", result.observation.finish_reason)
        print("system_error:", result.observation.system_error)


if __name__ == "__main__":
    asyncio.run(main())
