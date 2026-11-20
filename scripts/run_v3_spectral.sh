#!/usr/bin/env bash
# HI-JEPA v3 local-modal constrained readout fitting on frozen v2 features.
#
# Adds readout/K optimization (default 20k iterations, CPU FP64); source neural
# training is not repeated. No matrix or writes inside v1/v2 source directories.
# Usage: run_v3_spectral.sh [OUTPUT] [SOURCE_ROOT] [CLI flags ...]
# HIJEPA_V3_CONFIG selects a config path (relative paths use the caller's cwd).
# Default: configs/base_formal_v3_modal.json, with local-modal constraints.
# Frozen pre-modal V3: HIJEPA_V3_CONFIG=/path/to/configs/base_formal_v3.json
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_root}/runs/formal_v3_spectral}"
if [[ $# -gt 0 ]]; then shift; fi
source_root="${1:-${repo_root}/runs/formal_v2}"
if [[ $# -gt 0 ]]; then shift; fi
export HIJEPA_PYTHON="${HIJEPA_PYTHON:-python}"
device="${HIJEPA_V3_DEVICE:-cuda}"
config="${HIJEPA_V3_CONFIG:-${repo_root}/configs/base_formal_v3_modal.json}"
exec "${HIJEPA_PYTHON}" "${repo_root}/diagnostics/spectral_readout.py" \
    --config "${config}" \
    --source-root "${source_root}" \
    --output "${output}" \
    --device "${device}" \
    "$@"
