"""Verdict observer: reads assets and the runner's output, never agent code."""

import json
import os

env = os.environ.get
oracle = json.load(
    open(os.path.join(env("OPENENVD_ASSETS", "/assets"), "oracle", "oracle.json"))
)
runner = json.load(
    open(os.path.join(env("OPENENVD_INPUTS", "/inputs"), "runner", "report.json"))
)
verdict = {
    "score": 1.0 if oracle and not runner["assets_visible"] else 0.0,
    "runner_saw_assets": runner["assets_visible"],
}
with open(os.path.join(env("OPENENVD_OUT", "/out"), "verdict.json"), "w") as f:
    json.dump(verdict, f)
