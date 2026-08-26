#!/usr/bin/env bash
# Remove scratch created by this verification run. Never deletes EVIDENCE_ROOT.
set -euo pipefail

if [[ -z "${SCRATCH_ROOT:-}" ]]; then
  printf 'cleanup.sh needs SCRATCH_ROOT\n' >&2
  exit 2
fi

case "${SCRATCH_ROOT}" in
  /tmp/openenv-validate-verify-*)
    if [[ -d "${SCRATCH_ROOT}" ]]; then
      rm -rf "${SCRATCH_ROOT}"
      printf 'removed %s\n' "${SCRATCH_ROOT}"
    else
      printf 'no scratch at %s\n' "${SCRATCH_ROOT}"
    fi
    ;;
  *)
    printf 'refusing to remove unexpected SCRATCH_ROOT=%s\n' "${SCRATCH_ROOT}" >&2
    exit 2
    ;;
esac

if [[ -n "${EVIDENCE_ROOT:-}" ]]; then
  if [[ -d "${EVIDENCE_ROOT}" ]]; then
    printf 'evidence kept at %s\n' "${EVIDENCE_ROOT}"
  else
    printf 'warning: EVIDENCE_ROOT missing after cleanup: %s\n' "${EVIDENCE_ROOT}" >&2
  fi
fi
