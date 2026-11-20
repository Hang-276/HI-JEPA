#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_root}/runs/formal_v2}"
export HIJEPA_PYTHON="${HIJEPA_PYTHON:-python}"
export HIJEPA_MATRIX="${repo_root}/configs/matrix_baseline_v2.json"
export HIJEPA_PARITY_BUNDLE="${HIJEPA_PARITY_BUNDLE:-${repo_root}/runs/formal_v2_preflight/parity_bundle/bundle.pt}"
exec bash "${repo_root}/scripts/run_experiment.sh" e1 formal "${output}"
