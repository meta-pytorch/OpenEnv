"""Everything that must be true before the server accepts a request.

Four checks, in the order that fails cheapest first:

1. **LLM** — can it return token ids at all? A vLLM without
   `--return-tokens-as-token-ids --logprobs-mode processed_logprobs` answers every request perfectly
   well and returns no ids, so every rebuilt training row is empty and nothing reports an error. This
   is the one failure with no loud edge, so it is checked first and is fatal by default.
2. **sandbox credentials** — via Harbor's own `preflight()`, so the message names the exact missing
   variable rather than us guessing at one.
3. **datasets** — resolved and downloaded up front. A 2000-task repo takes real time to fetch, and a
   mistyped dataset name should fail here rather than on the first rollout.
4. **report** — print what is usable and, more usefully, what is not and why.

A caller gets the same `Capabilities` object the server serves over the wire, so what is printed at
startup and what a client can query are the same data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .capabilities import Capabilities, capabilities

# Load once, at import, so that Harbor's per-backend preflight sees the keys. Harbor reads
# credentials from the process environment and never from a file, so a `.env` that is not exported
# is invisible to it.
_ENV_LOADED = False


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Export `KEY=value` pairs from a dotenv file into the process environment.

    Existing variables win: an operator who exported something deliberately should not have it
    silently replaced by a checked-in file.

    Args:
        path (`str` or `Path`, *optional*):
            The file to read. Defaults to `$OPENENV_ENV_FILE`, then `./.env`.

    Returns:
        `list[str]`: Names of the variables that were set (values are never returned or logged).
    """
    global _ENV_LOADED
    candidate = Path(path or os.environ.get("OPENENV_ENV_FILE") or ".env").expanduser()
    if not candidate.is_file():
        return []

    applied: list[str] = []
    for raw in candidate.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    _ENV_LOADED = True
    return applied


def prepare(
    *,
    llm_url: str | None = None,
    model: str | None = None,
    datasets: list[str] | None = None,
    sandboxes: tuple[str, ...] | None = None,
    env_file: str | Path | None = None,
    require_llm: bool = True,
    quiet: bool = False,
) -> Capabilities:
    """Run every startup check and return what this server can do.

    Args:
        llm_url (`str`, *optional*):
            OpenAI-spec inference endpoint. No default: callers pass it explicitly.
        model (`str`, *optional*):
            Served model id. Defaults to `$OPENENV_MODEL`, else the LLM's only served model.
        datasets (`list[str]`, *optional*):
            Dataset specs to serve. Defaults to `$OPENENV_DATASETS` (comma-separated).
        sandboxes (`tuple[str, ...]`, *optional*):
            Backends to check. Defaults to `$OPENENV_SANDBOXES`, else the known set.
        env_file (`str` or `Path`, *optional*):
            Dotenv file to load before checking credentials.
        require_llm (`bool`, *optional*, defaults to `True`):
            Raise if the LLM cannot support capture. Set `False` to report and continue.
        quiet (`bool`, *optional*, defaults to `False`):
            Suppress the printed report.

    Returns:
        [`Capabilities`]: Harnesses, sandboxes, datasets and LLM status.

    Raises:
        RuntimeError: If `require_llm` and no URL was given, or the LLM cannot return token ids.
    """
    load_env_file(env_file)

    model = model or os.environ.get("OPENENV_MODEL", "")
    if datasets is None:
        raw = os.environ.get("OPENENV_DATASETS", "")
        datasets = [d.strip() for d in raw.split(",") if d.strip()]
    if sandboxes is None:
        raw = os.environ.get("OPENENV_SANDBOXES", "")
        sandboxes = tuple(s.strip() for s in raw.split(",") if s.strip()) or None

    if require_llm and not llm_url:
        raise RuntimeError(
            "no LLM URL given. Pass --llm-url (or llm_url=) explicitly; there is no default, "
            "because an unset endpoint yields rollouts that look fine and carry no token ids."
        )

    llm: dict[str, Any] = {}
    if llm_url:
        # Imported from the module rather than the package: the package re-exports a *function*
        # named `validate_llm`, which shadows the same-named submodule.
        from openenv.core.harness.capture.validate_llm import list_models, validate_llm

        # With no model given, ask the LLM what it serves. Convenient, and it also removes the
        # commonest startup mistake: guessing a short alias for a server that publishes its full
        # repo id.
        if not model:
            served = list_models(llm_url)
            model = served[0] if len(served) == 1 else ""

        report = validate_llm(llm_url, model) if model else None
        if report is None:
            llm = {
                "url": llm_url,
                "model": "",
                "ok": False,
                "findings": ["no model given and the LLM does not serve exactly one"],
            }
        else:
            llm = {
                "url": llm_url,
                "model": report.model,
                "ok": report.ok,
                "findings": report.findings,
                "served_models": report.served_models,
            }

    kwargs: dict[str, Any] = {"datasets": datasets, "llm": llm}
    if sandboxes:
        kwargs["sandboxes"] = sandboxes
    caps = capabilities(**kwargs)

    if not quiet:
        print(caps.render())

    if require_llm and llm and not llm.get("ok"):
        raise RuntimeError(
            "LLM cannot support token capture:\n  "
            + "\n  ".join(llm.get("findings") or ["unreachable"])
            + "\n\nStart vLLM with: --return-tokens-as-token-ids --logprobs-mode processed_logprobs"
        )
    return caps
