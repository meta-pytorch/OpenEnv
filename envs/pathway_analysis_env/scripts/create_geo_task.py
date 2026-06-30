#!/usr/bin/env python3
"""Create a GEO-style task case from counts + sample metadata.

Example:
PYTHONPATH=src:envs uv run python envs/pathway_analysis_env/scripts/create_geo_task.py \
  --task-id gseXXXX_example \
  --accession GSEXXXX \
  --summary "Short study summary" \
  --counts-file /path/to/counts.csv.gz \
  --metadata-csv /path/to/samples.csv \
  --reference-condition control \
  --alternate-condition treated
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_LIBRARIES = ["MSigDB_Hallmark_2020", "KEGG_2021_Human", "Reactome_2022"]


def _read_metadata_csv(path: Path) -> Tuple[List[str], Dict[str, str], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"sample_id", "condition"}
        missing = required - fields
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)} "
                "(required: sample_id, condition)"
            )

        sample_ids: List[str] = []
        sample_metadata: Dict[str, str] = {}
        conditions: List[str] = []
        seen_conditions = set()

        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            condition = (row.get("condition") or "").strip()
            if not sample_id or not condition:
                raise ValueError(
                    f"{path} has empty sample_id/condition row: {row!r}"
                )
            sample_ids.append(sample_id)
            sample_metadata[sample_id] = condition
            if condition not in seen_conditions:
                seen_conditions.add(condition)
                conditions.append(condition)

    if not sample_ids:
        raise ValueError(f"{path} has no sample rows")
    return sample_ids, sample_metadata, conditions


def _counts_dest_name(src: Path) -> str:
    name = src.name
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return name
    return f"{src.stem}.csv.gz" if src.suffix == ".gz" else f"{src.name}.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a GEO task case JSON from counts + sample metadata."
    )
    parser.add_argument("--task-id", required=True, help="Folder id under data/geo_eval/")
    parser.add_argument("--accession", required=True, help="Study accession (e.g. GSE216540)")
    parser.add_argument("--summary", required=True, help="Short human-readable study summary")
    parser.add_argument("--counts-file", type=Path, required=True, help="Path to counts .csv/.csv.gz")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="CSV with columns: sample_id,condition",
    )
    parser.add_argument("--reference-condition", required=True, help="Reference group name")
    parser.add_argument("--alternate-condition", required=True, help="Alternate group name")
    parser.add_argument(
        "--geo-ref-url",
        default="",
        help="Optional GEO URL; default is generated from accession",
    )
    parser.add_argument(
        "--libraries",
        default=",".join(DEFAULT_LIBRARIES),
        help="Comma-separated Enrichr libraries",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("envs/pathway_analysis_env/data/geo_eval"),
        help="Directory that stores task folders",
    )
    parser.add_argument(
        "--copy-counts",
        action="store_true",
        help="Copy counts file into task folder (default behavior)",
    )
    parser.add_argument(
        "--no-copy-counts",
        action="store_true",
        help="Do not copy counts file (use existing file under task folder)",
    )
    args = parser.parse_args()

    if args.reference_condition == args.alternate_condition:
        raise ValueError("reference-condition and alternate-condition must be different")

    sample_ids, sample_metadata, conditions = _read_metadata_csv(args.metadata_csv)
    if args.reference_condition not in conditions:
        raise ValueError(
            f"reference-condition '{args.reference_condition}' not found in metadata conditions {conditions}"
        )
    if args.alternate_condition not in conditions:
        raise ValueError(
            f"alternate-condition '{args.alternate_condition}' not found in metadata conditions {conditions}"
        )

    task_dir = args.out_dir / args.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    counts_src = args.counts_file.resolve()
    if not counts_src.exists():
        raise FileNotFoundError(f"counts file not found: {counts_src}")

    should_copy = not args.no_copy_counts
    if args.copy_counts:
        should_copy = True

    if should_copy:
        counts_name = _counts_dest_name(counts_src)
        counts_dst = task_dir / counts_name
        shutil.copy2(counts_src, counts_dst)
    else:
        counts_dst = counts_src
        if task_dir not in counts_dst.parents:
            raise ValueError(
                "--no-copy-counts requires counts-file to already be inside task folder"
            )

    counts_rel = f"geo_eval/{args.task_id}/{counts_dst.name}"
    case_name = f"{args.accession.lower()}_case.json"
    case_path = task_dir / case_name

    ref_url = args.geo_ref_url.strip() or f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={args.accession}"
    libraries = [s.strip() for s in args.libraries.split(",") if s.strip()]
    if not libraries:
        libraries = DEFAULT_LIBRARIES

    case = {
        "case_id": args.task_id,
        "strict_mode": False,
        "experiment_metadata": {
            "accession": args.accession,
            "reference": ref_url,
            "summary": args.summary,
        },
        "counts_file": counts_rel,
        "sample_ids": sample_ids,
        "sample_metadata": sample_metadata,
        "conditions": conditions,
        "default_contrast": {
            "reference": args.reference_condition,
            "alternate": args.alternate_condition,
        },
        "analysis_options": {
            "min_total_count": 10,
            "padj_alpha": 0.05,
            "de_query_direction": "both",
        },
        "enrichr_libraries": libraries,
        "true_pathway": "Unknown (GEO benchmark)",
    }
    case_path.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] wrote case: {case_path}")
    print(f"[ok] counts file: {counts_dst}")
    print(
        "[next] append to manifest:\n"
        "  PYTHONPATH=src:envs uv run python "
        "envs/pathway_analysis_env/scripts/append_task_to_manifest.py "
        f"--manifest envs/pathway_analysis_env/data/eval_manifest_geo3.json "
        f"--episode-id {args.task_id} --case-file {counts_rel.rsplit('/', 1)[0]}/{case_name} "
        '--hypothesis "your expected theme"'
    )


if __name__ == "__main__":
    main()

