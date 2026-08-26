#!/usr/bin/env bash
# Capture one `uv run openenv` invocation.
# Usage: validate-run.sh <feature-id> -- <openenv-args…>
# Requires REPO_ROOT and EVIDENCE_ROOT. Optional EXPECTED_EXIT.
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  printf 'usage: validate-run.sh <feature-id> -- <openenv-args…>\n' >&2
  exit 2
fi

feature_id="$1"
shift 2

if [[ -z "${REPO_ROOT:-}" || -z "${EVIDENCE_ROOT:-}" ]]; then
  printf 'validate-run.sh needs REPO_ROOT and EVIDENCE_ROOT\n' >&2
  exit 2
fi

out_dir="${EVIDENCE_ROOT}/${feature_id}"
mkdir -p "${out_dir}"

export PATH="${HOME}/.local/bin:${PATH}"

{
  printf 'cwd=%s\n' "${REPO_ROOT}"
  printf 'uv run openenv'
  for arg in "$@"; do
    printf ' %q' "${arg}"
  done
  printf '\n'
} >"${out_dir}/command.txt"

set +e
(
  cd "${REPO_ROOT}"
  uv run openenv "$@"
) >"${out_dir}/stdout.txt" 2>"${out_dir}/stderr.txt"
exit_code=$?
set -e

printf '%s\n' "${exit_code}" >"${out_dir}/exit_code.txt"

# If the user passed --output, copy the report next to the capture when it exists.
output_path=""
prev=""
for arg in "$@"; do
  if [[ "${prev}" == "--output" ]]; then
    output_path="${arg}"
  fi
  prev="${arg}"
done
if [[ -n "${output_path}" && -f "${output_path}" ]]; then
  cp "${output_path}" "${out_dir}/report.json"
fi

printf 'captured %s exit=%s\n' "${feature_id}" "${exit_code}"

if [[ -n "${EXPECTED_EXIT:-}" && "${exit_code}" != "${EXPECTED_EXIT}" ]]; then
  printf 'expected exit %s, got %s\n' "${EXPECTED_EXIT}" "${exit_code}" >&2
  exit 1
fi

exit 0
