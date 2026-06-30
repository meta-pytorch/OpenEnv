#!/usr/bin/env python3
"""Export agent-safe case JSON files (no ground-truth fields) under data/agent_safe/."""

from __future__ import annotations

import argparse
from pathlib import Path

from pathway_analysis_env.server.case_loader import export_agent_safe_case
from pathway_analysis_env.server.pathway_environment import DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "agent_safe",
        help="Output root (mirrors relative paths from data/).",
    )
    parser.add_argument(
        "cases",
        nargs="*",
        default=[
            "toy_case_001.json",
            "toy_case_002.json",
            "toy_case_legacy.json",
            "toy_case_no_default.json",
            "geo_eval/gse128911_mda_mb_134_vi_fulvestrant_vs_dmso/gse128911_case.json",
        ],
    )
    args = parser.parse_args()
    out_root: Path = args.out_dir
    for rel in args.cases:
        src = DATA_DIR / rel
        if not src.is_file():
            print(f"skip missing {src}")
            continue
        dst = out_root / rel
        export_agent_safe_case(src, dst)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
