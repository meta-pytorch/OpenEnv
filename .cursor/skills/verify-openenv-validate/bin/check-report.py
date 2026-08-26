#!/usr/bin/env python3
"""Parse a ValidationReport JSON file and print stable fields."""

from __future__ import annotations

import sys
from pathlib import Path

from openenv.validation.report import ValidationReport


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check-report.py <report.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    report = ValidationReport.model_validate_json(path.read_text())
    statuses = {item.check_id: item.status.value for item in report.results}
    print(f"schema={report.report_schema_version}")
    print(f"target={report.target}")
    print(f"signature={report.signature.value}")
    print(f"policy={report.policy_version}")
    print(f"verdict={report.verdict.value}")
    print(f"levels={[level.name.lower() for level in report.levels_run]}")
    print(f"static.manifest={statuses.get('static.manifest', 'absent')}")
    print(f"manifest={'present' if report.manifest is not None else 'null'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
