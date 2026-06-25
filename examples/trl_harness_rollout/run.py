"""Run the OpenEnv harness rollout worker end-to-end.

Two modes:
  --mode fake  : no GPU, no model. A scripted generator drives the real harness dynamic (agent owns
                 its loop, interception, multi-turn, verify). Good for laptops and CI.
  --mode vllm  : real generation with vLLM, capturing real token_ids + logprobs per turn.

Examples:
  python run.py --mode fake
  # GPU box, with `vllm serve <model> --port 8000` already running:
  python run.py --mode vllm --vllm-url http://localhost:8000 --model Qwen/Qwen2.5-3B-Instruct --dump captures.json
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import FakeGenerate, VLLMGenerate  # noqa: E402
from harness import TASKS, ProxyAgentSessionFactory  # noqa: E402
from interception import InterceptionProxy  # noqa: E402
from rollout_worker import HarnessRolloutWorker  # noqa: E402


def _wait_for_vllm(url: str, timeout_s: float = 60) -> None:
    import requests

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if requests.get(f"{url}/v1/models", timeout=5).status_code == 200:
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError(
        f"vLLM not reachable at {url}. Start it with: vllm serve <model> --port <port>"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fake", "vllm"], default="fake")
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument(
        "--dump",
        default=None,
        help="vllm mode: write per-turn token captures to this JSON",
    )
    p.add_argument("--max-workers", type=int, default=4)
    args = p.parse_args()

    if args.mode == "vllm":
        _wait_for_vllm(args.vllm_url)
        gen = VLLMGenerate(vllm_base_url=args.vllm_url, model=args.model)
        print(f"vLLM ready at {args.vllm_url}, model={args.model}")
        sampling = {"temperature": 0.7, "max_tokens": 256}
    else:
        gen = FakeGenerate()
        print("fake mode: no GPU, scripted generator driving the real harness dynamic")
        sampling = {"temperature": 1.0, "max_tokens": 256}

    proxy = InterceptionProxy()
    proxy.start()
    worker = HarnessRolloutWorker(
        session_factory=ProxyAgentSessionFactory(proxy),
        generate_api=gen,
        tasks=TASKS,
        max_turns=6,
        sampling=sampling,
    )

    print(f"\nRunning {len(TASKS)} rollouts concurrently...\n")
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            rollouts = [r for r in pool.map(worker.produce, TASKS) if r is not None]
    finally:
        proxy.stop()

    for r in rollouts:
        roles = [m["role"] for m in r.messages]
        print(
            f"  rollout {r.rollout_id}  reward={r.reward}  turns={int(r.metrics['turns'])}  roles={roles}"
        )
    solved = sum(1 for r in rollouts if r.reward > 0)
    print(f"\n  solved {solved}/{len(rollouts)}")
    if args.mode == "vllm":
        print(
            f"  real token capture (the data a TITO step would stitch): {gen.capture_summary()}"
        )
        if args.dump:
            n = gen.dump(args.dump)
            print(
                f"  dumped {n} per-turn captures (prompt_ids, completion_ids, logprobs) to {args.dump}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
