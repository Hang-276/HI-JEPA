#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${HIJEPA_PYTHON:-python}"
exec "${python_bin}" "${repo_root}/scripts/_experiment.py" "$@"
