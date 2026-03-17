#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, Sequence

import requests
from huggingface_hub import HfApi, get_token


def collect_space_ids(
    api: HfApi, namespace: str, suffixes: Sequence[str], explicit: Sequence[str]
) -> list[str]:
    suffixes = tuple(suffixes)
    spaces = list(api.list_spaces(author=namespace, limit=500))
    selected: list[str] = []

    for space in spaces:
        if any(space.id.endswith(suffix) for suffix in suffixes):
            selected.append(space.id)

    for space_id in explicit:
        if space_id not in selected:
            selected.append(space_id)

    return selected


def pick_domain_candidates(raw_domains: Iterable[dict]) -> list[dict]:
    ready = []
    fallback = []
    for domain in raw_domains:
        name = domain.get("domain")
        if not name:
            continue
        if domain.get("stage") == "READY":
            ready.append(domain)
        else:
            fallback.append(domain)
    return ready or fallback


def endpoint_ok(path: str, status_code: int | None) -> bool:
    if status_code is None:
        return False
    if path == "/mcp":
        return status_code in {200, 404}
    return status_code == 200


def probe_space(base_url: str, headers: dict, timeout: float = 20.0) -> list[dict]:
    endpoints = [
        ("GET", "/health"),
        ("GET", "/metadata"),
        ("GET", "/schema"),
        ("POST", "/mcp"),
    ]
    results = []

    session = requests.Session()
    for method, path in endpoints:
        full_url = base_url.rstrip("/") + path
        try:
            response = session.request(method, full_url, headers=headers, timeout=timeout)
            ok = endpoint_ok(path, response.status_code)
            results.append(
                {
                    "path": path,
                    "method": method,
                    "status": response.status_code,
                    "ok": ok,
                    "details": response.json()
                    if "application/json" in response.headers.get("Content-Type", "")
                    else response.text.strip(),
                }
            )
        except requests.RequestException as exc:
            results.append(
                {
                    "path": path,
                    "method": method,
                    "status": None,
                    "ok": False,
                    "details": f"{exc.__class__.__name__}: {exc}",
                }
            )

    return results


def stage_is_healthy(stage: str | None) -> bool:
    return bool(stage) and ("RUNNING" in stage or stage == "SLEEPING")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify private OpenEnv spaces by suffix/explicit IDs."
    )
    parser.add_argument(
        "--hf-namespace",
        default="openenv",
        help="Namespace owning the Spaces (default: openenv).",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=[],
        help="Suffix to match Spaces names (repeatable; defaults to -0.2.2 when omitted).",
    )
    parser.add_argument(
        "--space-id",
        action="append",
        default=[],
        help="Explicit Space id to include (repeatable).",
    )
    args = parser.parse_args()
    suffixes = args.suffix or ([] if args.space_id else ["-0.2.2"])

    token = get_token()
    if not token:
        print("error: no HuggingFace token found; run `hf auth login`.", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    space_ids = collect_space_ids(api, args.hf_namespace, suffixes, args.space_id)
    if not space_ids:
        print("error: no spaces found with the provided suffix/ids.", file=sys.stderr)
        sys.exit(1)

    verification: list[dict] = []
    headers = {"Authorization": f"Bearer {token}"}
    overall_success = True

    for sid in space_ids:
        info = api.space_info(sid, expand=["runtime"])
        runtime = getattr(info, "runtime", None)
        stage = getattr(runtime, "stage", None) if runtime else None
        domains = getattr(runtime, "raw", {}).get("domains", []) if runtime else []
        domain_candidates = pick_domain_candidates(domains) if domains else []
        error_message = getattr(runtime, "raw", {}).get("errorMessage") if runtime else None

        domain_attempts: list[dict] = []
        chosen_domain: str | None = None
        space_success = False

        if stage_is_healthy(stage) and domain_candidates:
            for candidate in domain_candidates:
                domain_name = candidate["domain"]
                base_url = f"https://{domain_name}"
                checks = probe_space(base_url, headers)
                success = all(check["ok"] for check in checks)
                domain_attempts.append(
                    {
                        "domain": domain_name,
                        "stage": candidate.get("stage"),
                        "success": success,
                        "checks": checks,
                    }
                )
                if success:
                    chosen_domain = domain_name
                    space_success = True
                    break

        overall_success = overall_success and space_success

        verification.append(
            {
                "space": sid,
                "stage": stage,
                "domain": chosen_domain
                or (domain_candidates[0]["domain"] if domain_candidates else None),
                "runtime_error": error_message,
                "checks": domain_attempts[-1]["checks"] if domain_attempts else [],
                "success": space_success,
                "domain_attempts": domain_attempts,
            }
        )

    print(json.dumps(verification, indent=2))
    if not overall_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
