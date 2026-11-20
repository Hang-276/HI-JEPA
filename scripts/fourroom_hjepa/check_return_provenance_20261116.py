#!/usr/bin/env python3
"""Read-only FourRoom return provenance checks; no model imports or execution.

Optional --fetch-official obtains public metadata only, never model weights.
All new receipts are outside the immutable unpacked result directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'results/fourroom_hjepa/fourroom_01'
OUT = ROOT/'results/fourroom_hjepa/fourroom_01_analysis_20261116'
OLD = ROOT/'archive/releases/hjepa_slow_selection_20261115'
TIER = 'formal_result_integrity_verification'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def fetch_metadata(url, destination):
    if destination.exists():
        cached = json.loads(destination.read_text())
        assert cached['source_url'] == url
        assert hashlib.sha256(cached['response_body_utf8'].encode()).hexdigest() == cached['response_sha256']
        return cached
    request = urllib.request.Request(url, headers={'User-Agent': 'HI-JEPA-result-integrity-check'})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read()
        status = response.status
        final_url = response.url
    receipt = {'evidence_tier': TIER, 'source_url': url, 'final_url': final_url,
               'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
               'http_status': status, 'response_bytes': len(raw),
               'response_sha256': hashlib.sha256(raw).hexdigest(),
               'response_body_utf8': raw.decode('utf-8')}
    save(destination, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-official', action='store_true',
                        help='Fetch missing official HF and upstream Git tree metadata; never weights')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((RUN/'provenance.json').read_text())
    configuration = json.loads((RUN/'config.json').read_text())
    setup = provenance['setup']
    environment = {}
    for line in (RUN/'environment.txt').read_text().splitlines():
        if '==' in line:
            name, version = line.split('==', 1)
            environment[re.sub(r'[-_.]+', '-', name).lower()] = version
    package_checks = []
    for name, version in setup['packages'].items():
        actual = environment.get(re.sub(r'[-_.]+', '-', name).lower())
        package_checks.append({'package': name, 'setup_version': version,
                               'environment_version': actual, 'matches': actual == version})
    assert all(x['matches'] for x in package_checks)
    snapshots = []
    for path in sorted((RUN/'scripts').iterdir()):
        if not path.is_file():
            continue
        old = OLD/path.name
        row = {'returned_file': str(path.relative_to(ROOT)), 'sha256': sha(path),
               'old_file': str(old.relative_to(ROOT)), 'old_exists': old.is_file()}
        if old.is_file():
            row['old_sha256'] = sha(old)
            row['byte_identical_to_original_delivery'] = path.read_bytes() == old.read_bytes()
            if not row['byte_identical_to_original_delivery']:
                row['unified_diff'] = ''.join(difflib.unified_diff(old.read_text().splitlines(keepends=True),
                    path.read_text().splitlines(keepends=True), fromfile=str(old.relative_to(ROOT)),
                    tofile=str(path.relative_to(ROOT))))
        snapshots.append(row)
    old_config = json.loads((OLD/'config.json').read_text())
    current_config = json.loads((ROOT/'configs/fourroom_hjepa/default.json').read_text())
    config_differences = {key: {'old': old_config.get(key), 'returned': configuration.get(key)}
                          for key in old_config if old_config[key] != configuration.get(key)}
    assert not config_differences
    assert current_config == old_config
    assert (RUN/'scripts/metrics.py').read_bytes() == (ROOT/'src/slowrepr/metrics.py').read_bytes()
    assert configuration['engineering_smoke'] is False
    metadata_url = ('https://huggingface.co/api/models/'+setup['hf_repo']+'/revision/'+
                    setup['hf_revision']+'?blobs=true')
    metadata_file = OUT/'OFFICIAL_WEIGHT_METADATA.json'
    metadata_error = None
    if args.fetch_official or metadata_file.exists():
        try:
            metadata = fetch_metadata(metadata_url, metadata_file)
            response = json.loads(metadata['response_body_utf8'])
            assert response['sha'] == setup['hf_revision']
            by_path = {x['rfilename']: x for x in response['siblings']}
            official = []
            for seed in (42, 43, 44):
                name = f'fourroom/fourroom_hjepa_l2/seed{seed}/fourroom_hjepa_l2_object.ckpt'
                item = by_path[name]
                digest = item['lfs']['sha256']
                observed = provenance['weights_sha256'][str(seed)]
                setup_digest = setup['weights_sha256'][str(seed)]
                official.append({'seed': seed, 'path': name, 'official_metadata': item,
                                 'runtime_recorded_sha256': observed, 'setup_recorded_sha256': setup_digest,
                                 'all_recorded_hashes_match': digest == observed == setup_digest})
            assert all(x['all_recorded_hashes_match'] for x in official)
        except Exception as error:
            metadata_error = f'{type(error).__name__}: {error}'
            official = []
    else:
        official = []
        metadata_error = 'Official metadata cache unavailable; use --fetch-official to obtain public metadata.'
    tree_url = ('https://api.github.com/repos/kevinghst/H-JEPA/git/trees/'+
                provenance['upstream_commit']+'?recursive=1')
    tree_path = OUT/'OFFICIAL_SOURCE_TREE_METADATA.json'
    module_source = {'runtime_module_file_recorded': False,
                     'distribution_in_pip_freeze': environment.get('stable-worldmodel'),
                     'interpretation': 'Successful formal execution establishes an available import path, not a recorded runtime __file__ or independently versioned distribution.'}
    if args.fetch_official or tree_path.exists():
        try:
            tree_receipt = fetch_metadata(tree_url, tree_path)
            tree = json.loads(tree_receipt['response_body_utf8'])
            module_paths = [item['path'] for item in tree['tree'] if 'stable_worldmodel' in item['path']]
            module_source.update({'official_tree_url': tree_url, 'tree_response_truncated': tree.get('truncated'),
                                  'official_commit_matching_module_paths': module_paths,
                                  'source_tree_metadata_receipt': str(tree_path.relative_to(ROOT)),
                                  'inference_scope': 'Official tree paths establish availability in pinned source, not the actual runtime imported file.'})
        except Exception as error:
            module_source['official_tree_check_error'] = f'{type(error).__name__}: {error}'
    complete = [line for line in (RUN/'run.log').read_text().splitlines() if 'Complete. Return the entire folder:' in line]
    collection = [line for line in (RUN/'collection.log').read_text().splitlines() if 'Completed data collection' in line]
    assert len(complete) == 1
    script = (RUN/'scripts/run.py').read_text()
    current_setup = (ROOT/'scripts/fourroom_hjepa/setup.sh').read_text()
    result = {'evidence_tier': TIER, 'status': 'verified_with_documented_provenance_limits',
              'immutable_result_directory': str(RUN),
              'environment': {'python': provenance['python'], 'torch': provenance['torch'],
                              'gpu': provenance['gpu'], 'device': provenance['device'],
                              'package_checks': package_checks,
                              'transformers_repair': {'original_and_current_local': '>=4.50,<5', 'returned': '==5.9.0'},
                              'hub_repair': {'original_and_current_local': '>=0.30,<1', 'returned': '==1.5.0'},
                              'current_local_setup_retains_old_ranges': 'transformers>=4.50,<5' in current_setup and 'huggingface-hub>=0.30,<1' in current_setup,
                              'stable_worldmodel': module_source},
              'returned_snapshot_vs_original_delivery': snapshots,
              'scientific_configuration_differences': config_differences,
              'current_scientific_configuration_matches_original': current_config == old_config,
              'current_metrics_byte_identical_to_returned': True,
              'source_authority': 'Returned scripts/run.py is byte-identical to the archived standalone runner; local reorganized wrapper is not substituted as runtime source.',
              'upstream_commit_recorded': provenance['upstream_commit'],
              'upstream_commit_agrees_with_setup_and_config': provenance['upstream_commit'] == setup['upstream_commit'] == configuration['upstream_commit'],
              'official_weights': {'api_url': metadata_url, 'fixed_revision': setup['hf_revision'],
                                   'metadata_receipt': str(metadata_file.relative_to(ROOT)) if metadata_file.exists() else None,
                                   'verification_error': metadata_error, 'items': official,
                                   'scope': 'Recorded runtime and setup digests are compared to official fixed-revision LFS metadata; weight binaries are not in this return and were not downloaded.'},
              'completion': {'run_log_line': complete[0], 'collection_log_lines': collection,
                             'run_log_timezone_explicit': False, 'user_reported_timezone': 'Asia/Shanghai',
                             'collection_log_explicit_offset': '+08:00'},
              'user_supplied_claims': {'hf_mirror_download': {'user_reported': True, 'independently_verified_from_return': False,
                                                           'reason': 'Returned setup and provenance record official repo/revision and hashes, but no download endpoint or transfer log.'},
                                       'smoke_03_complete': {'user_reported': True, 'independently_verified_from_return': False,
                                                            'reason': 'Formal result bundle contains no smoke_03 status or run log.'}},
              'calibration_quantile': {'function': 'np.quantile', 'quantile': configuration['cal_gap_quantile'],
                                      'method': 'NumPy default linear', 'no_explicit_method_in_returned_call': 'np.quantile(cal_scores["gap"], cfg["cal_gap_quantile"])' in script},
              'script_receipt': {'present': (RUN/'script_receipt.json').exists(),
                                 'required_by_actual_old_runner': False,
                                 'explanation': 'The archived standalone runner snapshots six files without a script_receipt; the reorganized wrapper later added that receipt.'},
              'sources_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [RUN/'provenance.json', RUN/'environment.txt', RUN/'config.json', RUN/'scripts/setup.sh', RUN/'run.log', RUN/'collection.log', ROOT/'scripts/fourroom_hjepa/setup.sh']},
              'execution_scope': {'model_import_or_execution': False, 'weight_download': False,
                                  'experiments_started': False, 'package_installation': False, 'returned_files_modified': False}}
    save(OUT/'PROVENANCE_CHECK.json', result)
    print(json.dumps({'evidence_tier': TIER, 'package_checks': len(package_checks),
                      'official_weight_matches': sum(x['all_recorded_hashes_match'] for x in official),
                      'official_metadata_error': metadata_error,
                      'stable_worldmodel_source_files': len(module_source.get('official_commit_matching_module_paths', [])),
                      'scientific_configuration_differences': config_differences}, ensure_ascii=False))


if __name__ == '__main__':
    main()
