#!/usr/bin/env python3
"""Read-only, prespecified diagnostics of the sealed FourRoom server return.

No model loading, fitting, simulation, threshold selection or new test data.
The bootstrap only resamples the existing shared TEST episode rows jointly.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TIER = 'exploratory_analysis_of_sealed_visual_results'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def top2(values):
    return np.sort(np.argsort(values, axis=1, kind='stable')[:, :2], axis=1)


def set_counts(sets):
    keys, counts = np.unique(sets, axis=0, return_counts=True)
    return {','.join(map(str, key)): int(count) for key, count in zip(keys, counts)}


def joint_counts(wrong1, wrong2, mask):
    n = int(mask.sum())
    return {'windows': n,
            'both_correct': int(np.count_nonzero(mask & ~wrong1 & ~wrong2)),
            'only_level1_correct': int(np.count_nonzero(mask & ~wrong1 & wrong2)),
            'only_level2_correct': int(np.count_nonzero(mask & wrong1 & ~wrong2)),
            'both_wrong': int(np.count_nonzero(mask & wrong1 & wrong2)),
            'level1_errors': int(np.count_nonzero(mask & wrong1)),
            'level2_errors': int(np.count_nonzero(mask & wrong2)),
            'level1_error_rate': float(wrong1[mask].mean()) if n else None,
            'level2_error_rate': float(wrong2[mask].mean()) if n else None}


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return {'n': int(values.size), 'mean': float(values.mean()) if values.size else None,
            'rms': float(np.sqrt(np.mean(values*values))) if values.size else None,
            'quantiles_05_50_95': np.quantile(values, [.05, .5, .95]).tolist() if values.size else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'results/fourroom_hjepa/fourroom_01')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/fourroom_hjepa/fourroom_01_analysis_20261116')
    args = parser.parse_args()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output == source or output.is_relative_to(source):
        parser.error('Diagnostics must stay outside the original sealed run directory')
    if not output.is_relative_to((ROOT/'results/fourroom_hjepa').resolve()):
        parser.error('Output must stay within results/fourroom_hjepa/')
    target = output/'DIAGNOSTICS.json'
    if target.exists():
        raise FileExistsError('Preserve the existing diagnostic receipt; choose a new output directory')
    started = time.perf_counter()
    sources = {}

    def load_json(relative):
        path = source/relative
        sources[relative] = sha(path)
        return json.loads(path.read_text())

    def load_npz(relative):
        path = source/relative
        sources[relative] = sha(path)
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    config = load_json('config.json')
    summary = load_json('summary.json')
    thresholds = load_json('thresholds.json')
    splits = load_json('splits.json')
    status = load_json('status.json')
    assert status['status'] == 'COMPLETE' and not summary['engineering_smoke']
    truth = load_npz('test_truth.npz')['states']
    reference = load_npz('reference_truth.npz')
    physical_variance = reference['variance']
    assert np.all(physical_variance > 0) and truth.shape[-1] == 4
    n = len(truth)
    assert len(splits['test']['episode_ids']) == n
    axes = summary['axes']
    bootstrap_seed, replicates = 2026111604, 1000
    rng = np.random.default_rng(bootstrap_seed)
    # Exactly one shared weight matrix for every seed, layer and intervention.
    # This is resampling existing episode rows, not trajectory generation.
    weights = rng.multinomial(n, np.full(n, 1/n), size=replicates).astype(float)

    def interval(values, mask):
        denominator = weights@mask.astype(float)
        numerator = weights@(values.astype(float)*mask)
        valid = denominator > 0
        return {'point': float(np.mean(values[mask])) if mask.any() else None,
                'percentile_95': np.quantile(numerator[valid]/denominator[valid], [.025, .975]).tolist() if valid.any() else None,
                'valid_bootstrap_replicates': int(valid.sum())}

    def diagnostic_comparison(original, revised, mask):
        count = int(mask.sum())
        repaired = int(np.count_nonzero(mask & original & ~revised))
        harmed = int(np.count_nonzero(mask & ~original & revised))
        improvement = original.astype(float)-revised.astype(float)
        return {'windows': count, 'original_errors': int(np.count_nonzero(original & mask)),
                'diagnostic_errors': int(np.count_nonzero(revised & mask)),
                'original_risk': float(original[mask].mean()) if count else None,
                'diagnostic_risk': float(revised[mask].mean()) if count else None,
                'repairs': repaired, 'harms': harmed, 'net_fewer_errors': repaired-harmed,
                'accuracy_gain': interval(improvement, mask)}

    cells = []
    runtime_cells = {}
    maximum_identity_error = 0.
    all_mask = np.ones(n, dtype=bool)
    for seed in config['model_seeds']:
        for level in (1, 2):
            name = f'seed{seed}_level{level}'
            audit = load_json(f'{name}/audit.json')
            prediction = load_npz(f'{name}/test_prediction.npz')['prediction']
            scores = load_npz(f'{name}/test_scores.npz')
            predicted_variance = load_npz(f'{name}/reference.npz')['variance']
            true_scores, pred_scores = scores['true_scores'], scores['predicted_scores']
            true_top = scores['true_top2']
            wrong = scores['wrong']
            assert np.array_equal(top2(true_scores), true_top)
            assert np.array_equal(np.any(top2(pred_scores) != true_top, axis=1), wrong)
            cross = scores['cross_term']/predicted_variance
            energy = scores['error_energy']/predicted_variance
            reference_term = true_scores*(physical_variance/predicted_variance-1)
            error = pred_scores-true_scores
            identity_error = float(np.max(np.abs(error-cross-energy-reference_term)))
            maximum_identity_error = max(maximum_identity_error, identity_error)
            assert identity_error < 1e-12
            state_mse = np.mean((prediction-truth)**2, axis=1)/physical_variance
            masks = {'own_cal': scores['gap'] >= thresholds['own_cal'][name],
                     'level1_cal': scores['gap'] >= thresholds['level1_cal'][f'seed{seed}']}
            for rule, mask in masks.items():
                row = next(r for r in summary['rows'] if r['cell'] == name and r['rule'] == rule)
                assert int(mask.sum()) == row['accepted'] and int(np.count_nonzero(wrong & mask)) == row['errors']
            variants = {
                'replace_predicted_REF_with_physical_REF': pred_scores*predicted_variance/physical_variance,
                'subtract_signed_cross_term': pred_scores-cross,
                'subtract_residual_increment_energy': pred_scores-energy,
            }
            diagnoses = {}
            for label, diagnostic_scores in variants.items():
                diagnostic_top2 = top2(diagnostic_scores)
                diagnostic_wrong = np.any(diagnostic_top2 != true_top, axis=1)
                diagnoses[label] = {'all_windows': diagnostic_comparison(wrong, diagnostic_wrong, all_mask),
                                    'original_acceptance_masks': {r: diagnostic_comparison(wrong, diagnostic_wrong, m) for r, m in masks.items()},
                                    'diagnostic_top2_counts': set_counts(diagnostic_top2),
                                    'negative_score_entries': int(np.count_nonzero(diagnostic_scores < 0)),
                                    'diagnostic_boundary_tie_windows': int(np.count_nonzero(np.diff(np.sort(diagnostic_scores, axis=1), axis=1)[:, 1] == 0))}
            selected = true_top
            full = np.broadcast_to(np.arange(4), (n, 4))
            outside = full[~np.any(full[:, :, None] == selected[:, None, :], axis=2)].reshape(n, 2)

            def pair_differences(array):
                return (np.take_along_axis(array, outside, axis=1)[:, None, :]
                        -np.take_along_axis(array, selected, axis=1)[:, :, None])

            pair_gap = pair_differences(true_scores)
            predicted_pair_gap = pair_differences(pred_scores)
            pair_parts = {'signed_cross': pair_differences(cross),
                          'residual_energy': pair_differences(energy),
                          'reference_scale': pair_differences(reference_term)}
            pair_error = pair_differences(error)
            assert np.max(np.abs(predicted_pair_gap-pair_gap-pair_error)) < 1e-12
            assert np.max(np.abs(pair_error-sum(pair_parts.values()))) < 1e-12
            predicted_tie = predicted_pair_gap == 0
            reversed_pair = (predicted_pair_gap < 0) | (predicted_tie & (outside[:, None, :] < selected[:, :, None]))
            assert np.array_equal(reversed_pair.any(axis=(1, 2)), wrong)
            positive_gap = pair_gap > 0
            pair_detail = []
            for selected_axis in range(4):
                for outside_axis in range(4):
                    if selected_axis == outside_axis:
                        continue
                    mask = ((selected[:, :, None] == selected_axis) & (outside[:, None, :] == outside_axis))
                    count = int(mask.sum())
                    if count:
                        pair_detail.append({'selected_axis': axes[selected_axis], 'outside_axis': axes[outside_axis], 'pairs': count,
                                            'reversed_pairs': int(np.count_nonzero(mask & reversed_pair)),
                                            'true_gap': quantiles(pair_gap[mask]), 'score_error_difference': quantiles(pair_error[mask]),
                                            'component_differences': {k: quantiles(v[mask]) for k, v in pair_parts.items()}})
            cell = {'cell': name, 'seed': seed, 'level': level, 'audit': audit,
                    'all_window_errors': int(wrong.sum()), 'all_window_error': float(wrong.mean()),
                    'all_window_error_bootstrap': interval(wrong, all_mask),
                    'true_top2_counts': set_counts(true_top),
                    'predicted_top2_counts': set_counts(scores['predicted_top2']),
                    'physical_reference_variance': physical_variance.tolist(),
                    'predicted_reference_variance': predicted_variance.tolist(),
                    'predicted_to_physical_reference_variance_ratio': (predicted_variance/physical_variance).tolist(),
                    'TEST_normalized_state_mse_by_axis': {a: quantiles(state_mse[:, i]) for i, a in enumerate(axes)},
                    'score_error_components_by_axis': {label: {a: quantiles(v[:, i]) for i, a in enumerate(axes)}
                                                      for label, v in [('total', error), ('signed_cross', cross), ('residual_energy', energy), ('reference_scale', reference_term)]},
                    'four_true_selected_vs_unselected_pairs': {
                        'pairs_per_window': 4, 'pair_count': int(pair_gap.size),
                        'true_zero_gap_pairs': int(np.count_nonzero(pair_gap == 0)),
                        'true_boundary_tie_windows': int(np.count_nonzero(scores['true_gap'] == 0)),
                        'predicted_zero_gap_pairs': int(np.count_nonzero(predicted_tie)),
                        'predicted_boundary_tie_windows': int(np.count_nonzero(scores['gap'] == 0)),
                        'reversed_pairs': int(reversed_pair.sum()), 'windows_with_any_reversal': int(wrong.sum()),
                        'true_pair_gap': quantiles(pair_gap), 'predicted_pair_gap': quantiles(predicted_pair_gap),
                        'absolute_total_error_over_positive_true_gap': quantiles(np.abs(pair_error[positive_gap])/pair_gap[positive_gap]),
                        'component_differences': {k: quantiles(v) for k, v in pair_parts.items()},
                        'component_absolute_over_positive_true_gap': {k: quantiles(np.abs(v[positive_gap])/pair_gap[positive_gap]) for k, v in pair_parts.items()},
                        'axis_pair_detail': pair_detail,
                        'statistical_unit_note': 'Four comparisons within one physical episode are not independent samples.'},
                    'oracle_diagnostics': diagnoses}
            cells.append(cell)
            runtime_cells[name] = {'wrong': wrong, 'masks': masks, 'state_mse': state_mse}
    paired = []
    for seed in config['model_seeds']:
        one, two = runtime_cells[f'seed{seed}_level1'], runtime_cells[f'seed{seed}_level2']
        delta = two['wrong'].astype(float)-one['wrong'].astype(float)
        record = {'seed': seed, 'all_windows': joint_counts(one['wrong'], two['wrong'], all_mask),
                  'all_error_difference_level2_minus_level1': interval(delta, all_mask),
                  'mean_normalized_TEST_MSE_difference_level2_minus_level1_by_axis': (two['state_mse']-one['state_mse']).mean(axis=0).tolist(),
                  'selection_rules': {}}
        for rule in ('own_cal', 'level1_cal'):
            m1, m2 = one['masks'][rule], two['masks'][rule]
            subsets = {'both_accepted': m1 & m2, 'only_level1_accepted': m1 & ~m2,
                       'only_level2_accepted': ~m1 & m2, 'neither_accepted': ~m1 & ~m2}
            record['selection_rules'][rule] = {
                'level1_original_acceptance': joint_counts(one['wrong'], two['wrong'], m1),
                'level2_original_acceptance': joint_counts(one['wrong'], two['wrong'], m2),
                'partition': {label: {**joint_counts(one['wrong'], two['wrong'], mask),
                                     'error_difference_level2_minus_level1': interval(delta, mask)}
                              for label, mask in subsets.items()}}
        paired.append(record)
    for relative, expected in sources.items():
        assert sha(source/relative) == expected, relative
    result = {'evidence_tier': TIER, 'status': 'completed_prespecified_read_only_diagnostics',
              'source_directory': str(source), 'axes': axes, 'TEST_episodes': n,
              'bootstrap': {'seed': bootstrap_seed, 'replicates': replicates,
                            'unit': 'existing TEST episode; same resampling weights for all seeds/layers/corrections',
                            'interval': 'descriptive percentile 95%, conditional on fixed trained models, readouts, REF and CAL; not simultaneous or confirmatory'},
              'score_error_identity': 'predicted_score - true_score = cross/predREF + residual_energy/predREF + true_score*(physicalREF/predREF - 1)',
              'maximum_score_decomposition_absolute_error': maximum_identity_error,
              'diagnostic_selection': 'Original true answers, thresholds, and original acceptance masks stay frozen. No corrected-score reselection is performed.',
              'cells': cells, 'paired_layers_by_seed': paired,
              'all_six_audit_cells_eligible': all(c['audit']['eligible'] for c in cells),
              'source_sha256': sources,
              'limitations': ['All six readouts fail recovery admission; these data do not establish reliable selection conditional on all factors being recovered.',
                              'AUDIT recovery levels differ across layers and axes; no matched-recovery causal attribution is made.',
                              'Poor probe recovery does not establish information absence in the encoder representation.',
                              'Oracle corrections use TEST truth and are diagnostic, not deployable methods or newly calibrated rules.',
                              'Component cancellation or oracle repair cannot by itself identify a new mechanism.',
                              'The same 2048 episodes underlie every model and layer; no pooling as six independent test sets.',
                              'No additional lag/condition search, model fit, threshold adjustment, or independent confirmation set was introduced.'],
              'elapsed_seconds': time.perf_counter()-started}
    output.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'evidence_tier': TIER, 'output': str(target), 'cells': len(cells),
                      'paired_seeds': len(paired), 'max_decomposition_error': maximum_identity_error,
                      'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
