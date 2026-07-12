# SPDX-License-Identifier: BSD-3-Clause

"""Fail-closed trust-boundary guards used by remote certification."""

from __future__ import annotations

from typing import Any

from openenv.core.containers.runtime.hf_sandbox_provider import (
    credential_environment_names,
)


def ensure_official_hf_sandbox(provider: Any) -> None:
    """Reject shared HF pools for official certification workloads."""
    if getattr(provider, "isolation_mode", None) != "dedicated":
        raise RuntimeError(
            "Official certification requires a dedicated Hugging Face sandbox; "
            "pooled execution is a same-user trust boundary."
        )
    if getattr(provider, "secrets", None):
        raise RuntimeError(
            "Official subject sandboxes cannot receive coordinator secrets."
        )
    env_vars = getattr(provider, "env_vars", None) or {}
    credential_names = credential_environment_names(env_vars)
    if credential_names:
        raise RuntimeError(
            "Official subject sandboxes cannot receive credential-like "
            f"environment variables: {', '.join(credential_names)}"
        )
    lock = getattr(provider, "_lock_for_official_certification", None)
    if not callable(lock):
        raise RuntimeError(
            "Official certification requires a provider that can lock its "
            "validated execution settings."
        )
    lock()
