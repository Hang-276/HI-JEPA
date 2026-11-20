"""Frozen-model transfer adapters; visible target CAL fitting and named-dose queries.

Historical physical axis correspondences are reused explicitly. Physical state
levels stay inside the simulator-side query adapter and the score-only reveal.
"""

import copy
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = ROOT / 'outputs/HCF_STABLE_ID_RUN_20261028_V2/confirmation_v2'
OLD_QUERY = HERE.parent / 'fine_local_dose_window_20261110/query_data.py'
ACTIVE = HERE.parent / 'fine_active_shift_20261109'
ACTIVE_FORECAST = ACTIVE / 'FORECAST.json'
ACTIVE_LOCK = ACTIVE / 'LOCK.json'
PRIMARY_LOADER = HERE.parent / 'fine_independent_validation_20261109/fine_forward.py'
EXTRA_LOADER = HERE.parent / 'fine_replication_extension_20261109/extra_fine_forward.py'
MATCH = HERE.parent / 'step6_fine_20261105/forecast_lock.json'
KEYS = ('eps05_seed1_to008', 'eps008_seed0_to05')
CONFIGS = {
    'eps05_seed1_to008': dict(slot='fhn_fresh_eps05', rep=1, source_epsilon=.5, target_epsilon=.08,
                             calibration_seed=2026160000, query_seed=2026175000),
    'eps008_seed0_to05': dict(slot='fhn_fresh_eps008', rep=0, source_epsilon=.08, target_epsilon=.5,
                             calibration_seed=2026159000, query_seed=2026174000),
}
DOSE = .0175
CAL_DOSE = .035
CAL_TIMES = (8, 24, 40, 56, 72, 88, 104, 120)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_query = _load('_fine_transfer_existing_query', OLD_QUERY)
prior, shift, base = _query.prior, _query.shift, _query.prior.ac.base


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _config(key):
    if key not in CONFIGS:
        raise ValueError('Use a predeclared transfer key')
    return CONFIGS[key]


@lru_cache(maxsize=2)
def _loader(slot):
    return _load('_fine_transfer_' + slot,
                 EXTRA_LOADER if slot == 'fhn_fresh_eps05' else PRIMARY_LOADER)


def _calibration_pairs(key):
    config = _config(key)
    frozen = _read(ACTIVE_FORECAST)
    assert frozen['lock_sha'] == sha(ACTIVE_LOCK)
    row = frozen['slots'][config['slot']]['panels'][0]
    receipt, expected = row['calibration'], row['active']
    assert (receipt['slot'], receipt['epsilon'], receipt['count'], receipt['role'], receipt['seed']) == (
        config['slot'], config['target_epsilon'], 96, 'calibration', config['calibration_seed'])
    assert expected['seed'] == config['query_seed'] and expected['contexts'] == list(CAL_TIMES)
    fold = np.arange(96) % 4
    pairs, audit = prior.ac.observed_pairs(receipt, fold == 0, fold == 1, CAL_TIMES, config['query_seed'])
    assert audit['hashes'] == expected['hashes']
    np.testing.assert_array_equal(pairs['fit']['shifts'][:4], CAL_DOSE * np.eye(4))
    np.testing.assert_array_equal(pairs['check']['shifts'], -DOSE * np.eye(4))
    return pairs, dict(calibration=receipt, active=audit, original_forecast_sha=sha(ACTIVE_FORECAST),
                       fit_trajectory_indices=np.flatnonzero(fold == 0).tolist(),
                       check_trajectory_indices=np.flatnonzero(fold == 1).tolist(),
                       fit_dose=CAL_DOSE, check_dose=-DOSE, fit_contexts=list(CAL_TIMES),
                       unused_fit_branch_dose=CAL_DOSE * np.sqrt(2),
                       physical_levels_used=False, original_query_hashes_match=True)


def _readout(encode, pairs):
    baseline = pairs['fit']['baseline']
    values = np.asarray(encode(np.concatenate([baseline[None], baseline[None],
                                              pairs['fit']['branches']], axis=0)), dtype=np.float64)
    duplicate = float(max(np.abs(values[0] - values[1]).max(),
                          np.abs(values[0] - encode(baseline)).max()))
    changes = values[2:6] - values[0]
    target = np.broadcast_to(CAL_DOSE * np.eye(4)[:, None, None], changes.shape)
    x, y = changes.reshape(-1, 4), target.reshape(-1, 4)
    xscale = np.sqrt((x**2).mean(axis=0))
    if not np.isfinite(x).all() or np.any(xscale <= 0):
        return None, dict(available=False, reason='invalid_calibration_changes', duplicate_error=duplicate)
    beta, _, rank, singular = np.linalg.lstsq(x / xscale, y, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if rank < 4 or condition is None or condition >= 1e6:
        return None, dict(available=False, reason='calibration_rank_condition', rank=int(rank),
                          condition=condition, duplicate_error=duplicate)
    beta /= xscale[:, None]
    check_base = encode(pairs['check']['baseline'])
    check_changes = encode(pairs['check']['branches']) - check_base[None]
    check_target = np.broadcast_to(-DOSE * np.eye(4)[:, None, None], check_changes.shape)
    return beta, dict(available=True, rank=int(rank), condition=condition, duplicate_error=duplicate,
                       beta=beta.tolist(), fit_relative_rmse=float(np.linalg.norm(x @ beta - y) / np.linalg.norm(y)),
                       check_relative_rmse=float(np.linalg.norm(check_changes @ beta - check_target) / np.linalg.norm(check_target)),
                       fitting_rule='positive_0.035_named_dose_OLS_with_column_RMS_scaling',
                       check_used_for_fitting=False)


@lru_cache(maxsize=2)
def open_model(key):
    """Return encoders, independently replayed target-CAL beta, and explicit provenance."""
    config = _config(key)
    loader = _loader(config['slot'])
    frozen = loader.load_fine(config['slot'], config['rep'])
    matches = _read(MATCH)['parents']
    correspondence = next(row for row in matches if row['slot'] == config['slot'] and row['rep'] == config['rep'])
    assert correspondence['assignment_objective_lead'] > 0
    permutation = np.asarray(correspondence['perm'], dtype=np.intp)
    np.testing.assert_array_equal(np.sort(permutation), np.arange(4))
    with np.load(ARCHIVE / 'data/TRAIN_CAL' / f"{config['slot']}.npz", allow_pickle=False) as archive:
        raw_train = archive['train__x'].astype(np.float64)
    raw_mean, raw_sd = raw_train.mean((0, 1)), raw_train.std((0, 1))
    assert np.all(raw_sd > 0)

    def encode(values):
        return ((frozen.encode(values).astype(np.float64) - frozen.mean) / frozen.std)[..., permutation]

    def raw_encode(values):
        return (np.asarray(values, dtype=np.float64) - raw_mean) / raw_sd

    pairs, calibration = _calibration_pairs(key)
    beta, fine_fit = _readout(encode, pairs)
    raw_beta, raw_fit = _readout(raw_encode, pairs)
    provenance = dict(key=key, **config, checkpoint=frozen.provenance,
                       normalization='archived Fine TRAIN G0 and raw public TRAIN moments',
                       historical_axis_correspondence=dict(perm=permutation.tolist(), source=str(MATCH),
                           sha256=sha(MATCH), assignment_objective_lead=correspondence['assignment_objective_lead'],
                           physical_matching_was_preexisting=True, new_target_truth_used=False),
                       calibration=calibration, fine_readout=fine_fit, raw_readout=raw_fit,
                       physical_levels_used_for_fit=False)
    return dict(encode=encode, beta=beta, available=beta is not None,
                raw=dict(encode=raw_encode, beta=raw_beta, available=raw_beta is not None),
                provenance=provenance, frozen=frozen, loader=loader)


def public_panel(key, count, seed):
    config = _config(key)
    return shift.fresh_observations(config['slot'], config['target_epsilon'], count, seed, 'evaluation')


def source_panel(key, count, seed):
    """Optional same-source visible control; not used or drawn by this adapter's smoke."""
    config = _config(key)
    return shift.fresh_observations(config['slot'], config['source_epsilon'], count, seed, 'evaluation')


def node_branches(key, public, receipt, flat_node_ids, dose=DOSE):
    config = _config(key)
    slot, epsilon = receipt['slot'], receipt['epsilon']
    if slot != config['slot'] or epsilon not in (config['source_epsilon'], config['target_epsilon']):
        raise ValueError('Query receipt belongs to a different frozen source mapping')
    root = shift._stream_seed(slot, receipt['seed'], receipt['role'])
    tag, count = shift.FRESH_TAG, receipt['count']
    owner, env, transform, physics, _ = shift._recipe(slot, epsilon, root)
    require = shift.base._require
    require(receipt['role'] == 'evaluation' and receipt['namespace'] == shift.FRESH_NAMESPACE
            and receipt['trajectory_seed'] == root and receipt['trajectory_tag'] == tag,
            'Unexpected transfer query receipt')

    def guard(cfg, t, n, capability):
        require(cfg == physics and t == tag and n == count and capability is None, 'Unexpected transfer replay')

    namespace = {'np': np, 'copy': copy, 'env': env, 'require': require,
                 'ah': owner.array_sha, '_draw_guard': guard}
    owner._extract(owner.FHN_SOURCE, ('_draw', '_physical'), namespace)
    physical, actions, regimes, streams = namespace['_physical'](physics, tag, count)
    mix = {'data': {'dimension': 4, 'data_seed': receipt['observation_mix_seed'],
                    'observation_layers': 4, 'observation_strength': .25}}

    def observe(state):
        return np.ascontiguousarray(transform(state, mix), dtype=np.float32)

    reproduced = dict(x=observe(physical), actions=actions, regimes=regimes)
    require(shift.base._identities(reproduced) == shift.base._identities(public) == receipt['public_arrays'],
            'Transfer public arrays mismatch')
    require(all(streams[name] == expected for name, expected in receipt['physical_rng_stream_hashes'].items()),
            'Transfer physical RNG receipt mismatch')
    ids = np.asarray(flat_node_ids)
    require(ids.ndim == 1 and ids.dtype.kind in 'iu' and len(ids) > 0
            and len(np.unique(ids)) == len(ids)
            and np.all((ids >= 0) & (ids < physical.shape[0] * physical.shape[1])), 'Invalid transfer node IDs')
    ids = ids.astype(np.int64, copy=False)
    require(np.isfinite(dose) and dose > 0, 'Dose must be positive and finite')
    points = physical.reshape(-1, 4)[ids, None, :]
    offsets = float(dose) * np.eye(4)
    plus, minus = observe(points + offsets), observe(points - offsets)
    return plus, minus, dict(public_parity=True, query_nodes=len(ids), physical_branches=8 * len(ids),
                            query_shape=list(plus.shape), node_ids_sha256=owner.array_sha(ids),
                            branch_sha256=dict(plus=owner.array_sha(plus), minus=owner.array_sha(minus)))


def endpoint_branches(key, public, receipt, selected, dose=DOSE):
    selected = np.asarray(selected)
    if selected.ndim != 1 or selected.dtype.kind not in 'iu' or len(np.unique(selected)) != len(selected):
        raise ValueError('Selected trajectory indices must be distinct integers')
    times = public['x'].shape[1]
    ids = (selected[:, None] * times + np.arange(times)).ravel()
    plus, minus, audit = node_branches(key, public, receipt, ids, dose)
    plus = plus.reshape(len(selected), times, 4, 4)
    minus = minus.reshape(len(selected), times, 4, 4)
    owner, _ = shift.base._sources()
    audit.update(query_shape=list(plus.shape),
                 branch_sha256=dict(plus=owner.array_sha(plus), minus=owner.array_sha(minus)))
    return plus, minus, audit


def replay_physical(receipt):
    """Score-only wrapper; no predictor/calibrator in this module calls it."""
    return shift.replay_physical(receipt)


def sources():
    paths = [Path(__file__), OLD_QUERY, ACTIVE_FORECAST, ACTIVE_LOCK, PRIMARY_LOADER, EXTRA_LOADER, MATCH]
    result = prior.source_hashes()
    for config in CONFIGS.values():
        branch = Path('representation/OursFine4') / config['slot'] / f"seed{config['rep']}"
        folder = ARCHIVE / 'execution/jobs' / branch
        completion = _read(folder / 'COMPLETED.json')
        paths.extend([folder / 'COMPLETED.json', folder / 'G0.json', folder / 'model/Fine_latest.pt',
                      folder / 'model/Fine.json', ARCHIVE / 'execution/registration' / branch / 'JOB.json',
                      ARCHIVE / 'data/TRAIN_CAL' / f"{config['slot']}.npz", Path(completion['features']['path'])])
        for path, digest in completion['source_sha256'].items():
            assert sha(path) == digest
            result[str(Path(path).resolve())] = digest
    result.update({str(path.resolve()): sha(path) for path in paths})
    return result


def engineering_check():
    """Only archived TRAIN and existing CAL/evaluation replays; no new formal draws."""
    checks = {}
    old_forecast = _read(ACTIVE_FORECAST)
    for key in KEYS:
        config = _config(key)
        model = open_model(key)
        smoke = model['loader'].smoke(config['slot'], config['rep'], model['frozen'])
        assert model['available'] and model['raw']['available']
        assert max(model['provenance'][name]['duplicate_error'] for name in ('fine_readout', 'raw_readout')) <= 1e-4
        old_receipt = old_forecast['slots'][config['slot']]['panels'][0]['evaluation']
        public, receipt = public_panel(key, old_receipt['count'], old_receipt['seed'])
        assert receipt == old_receipt
        selected = np.array([0, 2], dtype=np.int64)
        plus, minus, audit = endpoint_branches(key, public, receipt, selected, DOSE)
        node_ids = np.array([0, 3, public['x'].shape[1] * 2 + 5], dtype=np.int64)
        small_plus, small_minus, small_audit = node_branches(key, public, receipt, node_ids, DOSE)
        np.testing.assert_array_equal(small_plus, plus[[0, 0, 1], [0, 3, 5]])
        np.testing.assert_array_equal(small_minus, minus[[0, 0, 1], [0, 3, 5]])
        old_parity = None
        if config['slot'] == prior.SLOT and config['target_epsilon'] == prior.TARGET_EPS:
            old_plus, old_minus, old_audit = _query.endpoint_branches(public, receipt, selected, DOSE)
            np.testing.assert_array_equal(plus, old_plus)
            np.testing.assert_array_equal(minus, old_minus)
            assert audit['branch_sha256'] == old_audit['branch_sha256']
            old_parity = True
        checks[key] = dict(train_cpu_smoke=smoke, provenance=model['provenance'],
                           existing_evaluation_receipt=receipt, endpoint_query=audit,
                           restricted_query=small_audit, restricted_node_parity=True,
                           old_endpoint_exact_parity=old_parity, newly_generated_samples=0,
                           physical_levels_read_by_predictor=False)
    return checks


if __name__ == '__main__':
    destination = HERE / 'ADAPTER_ENGINEERING.json'
    checks = engineering_check()
    with destination.open('x') as stream:
        json.dump(dict(checks=checks, sources=sources()), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: dict(train_parity=value['train_cpu_smoke']['max_abs_in_g0_units_by_axis'],
                               condition=value['provenance']['fine_readout']['condition'],
                               fit_rmse=value['provenance']['fine_readout']['fit_relative_rmse'],
                               check_rmse=value['provenance']['fine_readout']['check_relative_rmse'],
                               endpoint_parity=value['old_endpoint_exact_parity']) for key, value in checks.items()}, indent=2))
