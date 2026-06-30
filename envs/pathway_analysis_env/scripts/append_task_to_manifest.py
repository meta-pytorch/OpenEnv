#!/usr/bin/env python3
"""Append or update one task episode in an eval manifest.

Example:
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/append_task_to_manifest.py \
  --manifest envs/pathway_analysis_env/data/eval_manifest_geo3.json \
  --episode-id geo_gseXXXX \
  --case-file geo_eval/gseXXXX_example/gsexxxx_case.json \
  --hypothesis "estrogen response"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _default_manifest() -> Dict[str, Any]:
    return {
        "description": "GEO evaluation tasks.",
        "defaults": {
            "eval_mode": True,
            "max_steps": 30,
            "orchestrator_mode": True,
        },
        "episodes": [],
    }


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_manifest()
    return json.loads(path.read_text(encoding="utf-8"))


def _upsert_episode(episodes: List[Dict[str, Any]], episode: Dict[str, Any]) -> str:
    episode_id = episode["id"]
    for i, existing in enumerate(episodes):
        if existing.get("id") == episode_id:
            episodes[i] = episode
            return "updated"
    episodes.append(episode)
    return "added"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append or update a single episode in a manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--case-file", required=True, help="Path relative to data/ (e.g. geo_eval/.../case.json)")
    parser.add_argument("--hypothesis", default="pathway hypothesis")
    parser.add_argument(
        "--requires-pydeseq2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--requires-gseapy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--score-mode", default="keywords")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    episodes = manifest.setdefault("episodes", [])

    episode = {
        "id": args.episode_id,
        "case_file": args.case_file,
        "hypothesis": args.hypothesis,
        "requires_pydeseq2": bool(args.requires_pydeseq2),
        "requires_gseapy": bool(args.requires_gseapy),
        "score_mode": args.score_mode,
    }
    action = _upsert_episode(episodes, episode)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] {action} episode '{args.episode_id}' in {args.manifest}")
    print(f"[ok] total episodes: {len(episodes)}")


if __name__ == "__main__":
    main()

