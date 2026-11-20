#!/usr/bin/env python3
"""Read-only SHA-256 audit against the baseline saved before sandbox runs."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/exploratory_20261116'
TIER = 'exploratory_numerical_sandbox'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('/tmp/hijepa_exploratory_before.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    before_bytes = args.baseline.read_bytes()
    before = json.loads(before_bytes)
    previous = {r['path']: r for r in before['files']}
    paths = {str(p.relative_to(ROOT)): p for folder in before['folders']
             for p in (ROOT/folder).rglob('*') if p.is_file()}
    for name in previous:
        if (ROOT/name).is_file():
            paths[name] = ROOT/name
    rows = [{'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)}
            for name, path in sorted(paths.items())]
    current = {r['path']: r for r in rows}
    changed = [name for name in previous.keys() & current.keys() if previous[name] != current[name]]
    missing = sorted(previous.keys()-current.keys())
    added = sorted(current.keys()-previous.keys())
    duration = time.perf_counter()-started
    assert not changed and not missing and not added, (changed, missing, added)
    (OUT/'protected_before.json').write_bytes(before_bytes)
    (OUT/'protected_after.json').write_text(json.dumps({'evidence_tier': TIER, 'folders': before['folders'], 'files': rows}, ensure_ascii=False, indent=2)+'\n')
    python311_checked = []
    for path in sorted((ROOT/'scripts/exploratory_20261116').glob('*.py')):
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))
        python311_checked.append(str(path.relative_to(ROOT)))
    tier_checked = []
    for path in sorted(OUT.rglob('*')):
        if path.suffix == '.json':
            data = json.loads(path.read_text())
            assert next(iter(data)) == 'evidence_tier' and data['evidence_tier'] == TIER, path
            tier_checked.append(str(path.relative_to(ROOT)))
        elif path.suffix == '.md':
            assert path.read_text().splitlines()[0] == 'evidence_tier: '+TIER, path
            tier_checked.append(str(path.relative_to(ROOT)))
    report = ROOT/'docs/EXPLORATORY_MECHANISM_PROBES_20261116.md'
    assert report.read_text().splitlines()[0] == 'evidence_tier: '+TIER
    env = dict(os.environ)
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env.get('PATH', '')
    begin = time.perf_counter()
    check = subprocess.run(['python3', 'scripts/check_repository.py'], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    repo_seconds = time.perf_counter()-begin
    assert check.returncode == 0, check.stderr+check.stdout
    result = {'evidence_tier': TIER, 'status': 'passed', 'baseline_sha256': hashlib.sha256(before_bytes).hexdigest(),
              'protected_folders': before['folders'], 'protected_files': len(rows),
              'protected_bytes': sum(r['bytes'] for r in rows),
              'changed': changed, 'missing': missing, 'added': added, 'hash_scan_seconds': duration,
              'repository_check': {'command': 'python3 scripts/check_repository.py',
                                   'python3_directory': str(Path(sys.executable).parent),
                                   'seconds': repo_seconds, 'returncode': check.returncode,
                                   'result': json.loads(check.stdout)},
              'python311_syntax_checked': python311_checked,
              'evidence_tier_checked': tier_checked+[str(report.relative_to(ROOT))],
              'scope': 'Pre-existing worktree changes are not reset. Zero hash changes applies to the enumerated protected baseline, including predecessor theory document and theory checks.'}
    (OUT/'INTEGRITY.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('evidence_tier_checked', 'python311_syntax_checked')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
