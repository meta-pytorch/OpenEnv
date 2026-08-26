#!/usr/bin/env bash
# Read-only readiness for verify-openenv-validate. No args. Exit 0 if worth driving.
set -euo pipefail

fail() {
  printf 'not ok  %s\n' "$1" >&2
  exit 1
}

ok() {
  printf 'ok  %s\n' "$1"
}

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || true)}"
if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}/envs/echo_env" ]]; then
  fail "REPO_ROOT is not an OpenEnv checkout (set REPO_ROOT or run from the repo)"
fi

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  fail "uv not on PATH"
fi
ok "uv $(command -v uv)"

echo_yaml="${REPO_ROOT}/envs/echo_env/openenv.yaml"
if [[ ! -f "${echo_yaml}" ]]; then
  fail "missing ${echo_yaml}"
fi
if ! grep -q '^validation:' "${echo_yaml}"; then
  fail "${echo_yaml} has no validation: block"
fi
ok "envs/echo_env/openenv.yaml has validation:"

cd "${REPO_ROOT}"
if ! uv run openenv validate --help >/tmp/openenv-validate-help.$$ 2>&1; then
  fail "uv run openenv validate --help failed"
fi
if ! grep -q -- '--level' /tmp/openenv-validate-help.$$; then
  rm -f /tmp/openenv-validate-help.$$
  fail "validate --help missing --level"
fi
if ! grep -q -- '--skip-build' /tmp/openenv-validate-help.$$; then
  rm -f /tmp/openenv-validate-help.$$
  fail "validate --help missing --skip-build"
fi
rm -f /tmp/openenv-validate-help.$$
ok "openenv validate --help"

if ! uv run python -c "from openenv.validation.report import ValidationReport" >/dev/null 2>&1; then
  fail "ValidationReport import failed"
fi
ok "ValidationReport import"

printf 'doctor: ready\n'
