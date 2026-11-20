#!/usr/bin/env bash
# Fetch the existing formal result bundle; never launch or repeat an experiment.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    printf '%s\n' 'Usage: bash scripts/fourroom_hjepa/fetch_results_20261116.sh' \
        'Run on the Mac. SSH may ask for the server password in the terminal.' \
        'Uses HI-JEPA1; downloads and verifies the archive, without extracting it.'
    exit 0
fi
if (( $# != 0 )); then
    printf '%s\n' 'Unexpected arguments; use --help.' >&2
    exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
destination="$repo/archive/server_results/20261116"
archive="$destination/fourroom_01_results.tar.gz"
receipt="$destination/fourroom_01_results.transfer.json"
remote_archive=/root/autodl-tmp/hjepa-slow-selection/fourroom_01_results.tar.gz

if [[ -e "$archive" ]]; then
    printf '%s\n' 'An archive already exists; preserved without replacement:' "$archive"
    shasum -a 256 "$archive"
    exit 0
fi

mkdir -p "$destination"
transfer_tmp="$(mktemp -d "$destination/.fourroom-transfer.XXXXXX")"
socket_tmp="$(mktemp -d "${TMPDIR:-/tmp}/hjepa-ssh.XXXXXX")"
ssh_options=(-o ControlMaster=auto -o ControlPersist=120 -o "ControlPath=$socket_tmp/control" -o ConnectTimeout=15)
cleanup() {
    ssh "${ssh_options[@]}" -O exit HI-JEPA1 >/dev/null 2>&1 || true
    rm -f -- "$socket_tmp/control"
    rmdir -- "$socket_tmp" 2>/dev/null || true
    # Preserve any interrupted download and logs for inspection.
}
trap cleanup EXIT

printf '%s\n' 'Connecting to HI-JEPA1. Enter the server password here if requested.'
ssh "${ssh_options[@]}" HI-JEPA1 'set -eu
cd /root/autodl-tmp/hjepa-slow-selection
if [ ! -f fourroom_01_results.tar.gz ]; then
    tar -czf fourroom_01_results.tar.gz -C results fourroom_01
fi
sha256sum fourroom_01_results.tar.gz' > "$transfer_tmp/remote_before.sha256"

read -r expected_before ignored < "$transfer_tmp/remote_before.sha256"
if [[ ! "$expected_before" =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Server did not return a valid SHA256; stopped.' >&2
    exit 1
fi

scp "${ssh_options[@]}" "HI-JEPA1:$remote_archive" "$transfer_tmp/fourroom_01_results.tar.gz.partial"
ssh "${ssh_options[@]}" HI-JEPA1 \
    'sha256sum /root/autodl-tmp/hjepa-slow-selection/fourroom_01_results.tar.gz' \
    > "$transfer_tmp/remote_after.sha256"
read -r expected_after ignored < "$transfer_tmp/remote_after.sha256"
actual="$(shasum -a 256 "$transfer_tmp/fourroom_01_results.tar.gz.partial")"
actual="${actual%% *}"
if [[ "$expected_before" != "$expected_after" || "$actual" != "$expected_before" ]]; then
    printf '%s\n' 'SHA256 mismatch. Partial download preserved; no archive was published.' >&2
    exit 1
fi

# A hard link publishes the completed file atomically and refuses replacement.
ln "$transfer_tmp/fourroom_01_results.tar.gz.partial" "$archive"
python3 - "$receipt" "$archive" "$expected_before" "$expected_after" "$actual" <<'PY'
import datetime
import json
from pathlib import Path
import sys
receipt, archive, before, after, local = sys.argv[1:]
data = {
    "evidence_tier": "server_result_transfer_verification",
    "status": "downloaded_and_sha256_verified_not_yet_extracted",
    "ssh_alias": "HI-JEPA1",
    "server_archive": "/root/autodl-tmp/hjepa-slow-selection/fourroom_01_results.tar.gz",
    "local_archive": archive,
    "bytes": Path(archive).stat().st_size,
    "server_sha256_before": before,
    "server_sha256_after": after,
    "local_sha256": local,
    "all_three_hashes_match": before == after == local,
    "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "server_data_deleted": False,
    "experiment_launched": False,
}
with Path(receipt).open("x") as stream:
    json.dump(data, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
PY
rm -- "$transfer_tmp/fourroom_01_results.tar.gz.partial" \
    "$transfer_tmp/remote_before.sha256" "$transfer_tmp/remote_after.sha256"
rmdir -- "$transfer_tmp"
printf '%s\n' 'Downloaded and verified. Original server data retained.' "$archive" "SHA256: $actual" \
    'Tell Codex the download finished; it will inspect and archive the complete contents.'
