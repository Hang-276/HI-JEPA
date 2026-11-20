"""Read JSON schemas and existing summary scalars only; never import model/data code."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2/execution/jobs/representation/OursFine4'
SLOTS = ('ar_rich_fresh1', 'ar_rich_fresh2', 'fhn_fresh_eps008', 'fhn_fresh_eps05')
STEPS = (1500, 3000, 4500, 6000, 12000, 18000, 24000)
inputs = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    assert path.suffix == '.json'
    data = json.loads(path.read_text())
    inputs.append({'path': str(path), 'sha256': sha(path)})
    return data

def paths(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f'{prefix}.{k}' if prefix else k
            yield path
            yield from paths(v, path)
    elif isinstance(obj, list):
        for item in obj:
            yield from paths(item, prefix + '[]')

diagnostics_top, split_keys, regime_keys, bound_keys = set(), set(), set(), set()
fine_top, g0_top, feature_like_keys = set(), set(), set()
parents = []
diag_count = 0
posterior_key_hits = []
terms = ('posterior', 'logits', 'log_odds', 'calibration_bin', 'calibration_error', 'cross_entropy')
for slot in SLOTS:
    for rep in range(3):
        directory = BASE / slot / f'seed{rep}'
        fine = read(directory / 'model/Fine.json')
        g0 = read(directory / 'G0.json')
        fine_top.add(tuple(sorted(fine)))
        g0_top.add(tuple(sorted(g0)))
        assert fine['status'] == 'completed'
        assert tuple(fine['identity']['absolute_CAL_snapshots']) == STEPS
        assert len(fine['history']) == 7
        assert fine['best']['step'] in STEPS
        assert g0['horizons'] == [4, 16, 32]
        assert g0['physical_labels_used'] is False
        assert g0['selection_estimand'] == 'adaptive_unclipped_TRAIN_slowness_not_population_certificate'
        pair_hashes = {s: set() for s in ('TRAIN', 'CAL')}
        sample_sizes = {s: set() for s in ('TRAIN', 'CAL')}
        trajectories = {s: set() for s in ('TRAIN', 'CAL')}
        for step in STEPS:
            path = directory / 'model/diagnostics' / f'step{step:06d}.json'
            diagnostic = read(path)
            assert diagnostic['step'] == step
            diagnostics_top.add(tuple(sorted(diagnostic)))
            bound_keys.add(tuple(sorted(diagnostic['bounds'])))
            assert diagnostic['factor_labels_used'] is False
            for key in paths(diagnostic):
                if any(term in key.lower() for term in terms):
                    posterior_key_hits.append({'path': str(path), 'key': key})
            for split in ('TRAIN', 'CAL'):
                d = diagnostic[split]
                split_keys.add(tuple(sorted(d)))
                pair_hashes[split].add(d['pair_indices_sha256'])
                sample_sizes[split].add(d['pairs'])
                trajectories[split].add(len(d['per_trajectory']))
                assert set(d['standardized_innovation_by_regime']) == set(map(str, range(9)))
                for u in d['standardized_innovation_by_regime'].values():
                    regime_keys.add(tuple(sorted(u)))
                assert len(d['residual_vs_currentFine_action_correlation']) == 4
                assert all(len(row) == 6 for row in d['residual_vs_currentFine_action_correlation'])
                assert d['interpretation'] == 'Finite sampled diagnostics; not an axis-identification or uniform calibration certificate'
            diag_count += 1
        assert all(len(v) == 1 for v in pair_hashes.values())
        assert sample_sizes == {'TRAIN': {2048}, 'CAL': {2048}}
        parents.append({'slot': slot, 'rep': rep, 'selected_step': fine['best']['step'],
                        'snapshots': 7, 'pair_indices_shared_across_snapshots': True,
                        'pair_count_per_split_snapshot': 2048,
                        'source_trajectories_per_split': {s: sorted(v) for s, v in trajectories.items()},
                        'CAL_used_for_selection': True})
assert diag_count == 84
assert not posterior_key_hits
assert all(len(s) == 1 for s in (diagnostics_top, split_keys, regime_keys, bound_keys, fine_top, g0_top))

chain = read(ROOT / 'reports/stage_a_plus/round3/CHAINED_SCORE_AUDIT.json')
assert len(chain['scenarios']) == 24 and sum(len(s['pairs_all4']) for s in chain['scenarios']) == 96
summaries, all_scenarios = {}, []
for split in ('train', 'cal'):
    scenarios = [s for s in chain['scenarios'] if s['split'] == split]
    assert len(scenarios) == 12
    count = {'CS': 0, 'corr': 0, 'exact': 0}
    pair_count = {'CS': 0, 'corr': 0}
    for s in scenarios:
        pairs = s['pairs_all4']
        cs = min(p['chained_actual_lower_CS'] for p in pairs)
        corr = min(p['chained_actual_lower_corr'] for p in pairs)
        exact = min(p['exact_actual_gap_saved'] for p in pairs)
        for k, value in [('CS', cs), ('corr', corr), ('exact', exact)]:
            count[k] += value > 0
        for p in pairs:
            assert abs(p['chained_actual_lower_CS'] - (p['affine_lower_bound'] - p['A_to_actual_CS_measured_Q0'])) < 1e-12
            assert abs(p['chained_actual_lower_corr'] - (p['affine_lower_bound'] - p['A_to_actual_truth_correlation_triangle'])) < 1e-12
            assert p['exact_actual_gap_saved'] + 1e-12 >= max(p['chained_actual_lower_CS'], p['chained_actual_lower_corr'])
            pair_count['CS'] += p['chained_actual_lower_CS'] > 0
            pair_count['corr'] += p['chained_actual_lower_corr'] > 0
        assert (cs > 0) == s['all4_positive_chained_CS']
        assert (corr > 0) == s['all4_positive_chained_corr']
        all_scenarios.append({'slot': s['slot'], 'rep': s['rep'], 'split': split,
                              'minimum_chained_CS_lower': cs, 'minimum_chained_corr_lower': corr,
                              'minimum_exact_actual_gap_saved': exact,
                              'oracle_diagnostic_only': True})
    assert count['CS'] == chain['summary'][split]['all4_positive_chained_CS']
    assert count['corr'] == chain['summary'][split]['all4_positive_chained_corr']
    assert pair_count['CS'] == chain['summary'][split]['pair_positive_CS']
    assert pair_count['corr'] == chain['summary'][split]['pair_positive_corr']
    summaries[split] = {'parents': 12, 'physical_instances': 4, 'parent_success': count, 'pair_success': pair_count}
assert summaries['train']['parent_success'] == {'CS': 9, 'corr': 12, 'exact': 12}
assert summaries['cal']['parent_success'] == {'CS': 8, 'corr': 9, 'exact': 9}

norm = read(ROOT / 'reports/stage_a_plus/round5/ORACLE_GATE_FLOOR_AUDIT.json')
assert norm['summary']['common_eta_gate_numeric_oracle_veto_count'] == 6
assert norm['summary']['axis_specific_same_bound_numeric_oracle_veto_count'] == 3

source_names = [
    'work/hcf_method_freeze_20261025/fine_development.py',
    'work/hcf_stable_id_20261028/confirmation_v2/ours_bridge.py',
    'work/hcf_method_finalize_20261027/v1/explicit_selector.py',
]
sources = [{'path': str(ROOT / f), 'sha256': sha(ROOT / f)} for f in source_names]
diagnostic_source = (ROOT / source_names[0]).read_text()
assert 'mu, logstd = model.transition(y, a[r, t], u[r, t])' in diagnostic_source
assert "if best is None or item['CAL_NLL'] < best['CAL_NLL']:" in diagnostic_source
bridge_source = (ROOT / source_names[1]).read_text()
assert "z={split:fd.core.features(model,data[split]['x'],batch=2048) for split in data}" in bridge_source

result = {
    'scope': 'Specified confirmation_v2 Fine metadata schemas, owning logging source and old JSON-only oracle diagnostics; not an exhaustive repository absence claim or new scientific validation',
    'scope_sha256': sha(OUT / 'CURRENT_INPUT_SCOPE.md'),
    'script_sha256': sha(Path(__file__)),
    'metadata_counts': {'parents': 12, 'physical_instances': 4, 'diagnostics': 84, 'Fine_JSON': 12, 'G0_JSON': 12},
    'schema': {k: list(next(iter(v))) for k, v in [('diagnostics_top', diagnostics_top), ('split', split_keys), ('per_regime', regime_keys), ('bounds', bound_keys), ('Fine_top', fine_top), ('G0_top', g0_top)]},
    'posterior_related_key_search': {'terms': terms, 'matches': posterior_key_hits, 'interpretation': 'Supported by complete schema enumeration and owner diagnostic/feature-source review; no full counterfactual-u posterior/logits saved in these reviewed artifacts'},
    'parents': parents,
    'existing_oracle_chain_recount': summaries,
    'existing_oracle_chain_all_scenarios': all_scenarios,
    'round5_existing_norm_veto_summary': norm['summary'],
    'current_actual_Fine_truth_free_certificates_from_reviewed_inputs': 0,
    'unestablished_inputs': ['full-regime posterior or loss-difference records', 'Bayes risk or certified posterior excess upper bound', 'conditional-bin approximation/regularity and coverage bound', 'fixed-context statistical coverage', 'learned contrast singular value', 'anchor head odds/tail/output covariance confidence bounds', 'Q0/anchor density-ratio moment and endpoint coverage bounds', 'truth-containing complete compatible class', 'uniform signed candidate-to-Fine score correction intervals', 'population G0 score and variance estimation bounds'],
    'inputs': inputs,
    'owner_source_inputs': sources,
    'cost_per_execution': {'metadata_or_existing_summary_JSON_files_read': len(inputs), 'scientific_array_reads': 0, 'checkpoint_loads': 0, 'model_forwards': 0, 'new_fits': 0, 'random_draws': 0, 'new_physical_trajectories': 0, 'training': 0, 'server_connections': 0, 'GPU_jobs': 0},
    'all_checks_pass': True,
}
dest = OUT / 'CURRENT_INPUT_AUDIT.json'
dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'metadata_counts': result['metadata_counts'], 'oracle_recount': summaries, 'actual_unlabeled_certificates': 0}, ensure_ascii=False, indent=2))
