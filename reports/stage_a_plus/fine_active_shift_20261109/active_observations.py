"""Return shifted observation pairs and public displacement labels only."""
import importlib.util
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parent.parent / 'fine_drift_moment_20261109/paired_shift.py'
spec = importlib.util.spec_from_file_location('active_contexts', SOURCE)
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)
shift = context.shift


def sources():
    return context.source_hashes() | {str(Path(__file__).resolve()): shift.base._sha(Path(__file__))}


def observed_pairs(receipt, fit, check, times, seed):
    """Generate on the calibration side. State levels never leave this function."""
    slot, role = receipt['slot'], receipt['role']
    root = shift._stream_seed(slot, receipt['seed'], role)
    assert role == 'calibration' and receipt['namespace'] == shift.FRESH_NAMESPACE
    assert receipt['trajectory_seed'] == root and receipt['trajectory_tag'] == shift.FRESH_TAG
    assert receipt['observation_mix_seed'] == shift.base._old_seed(slot, 'mix')
    assert not np.any(fit & check) and int(fit.sum()) == int(check.sum()) == 24
    _, _, center, noise, _, transform, mix, error = context._exact_contexts(
        slot, receipt['epsilon'], receipt['count'], root, shift.FRESH_TAG, np.asarray(times),
        receipt['public_arrays'], receipt['physical_rng_stream_hashes'])
    owner, env, _, cfg, _ = shift._recipe(slot, receipt['epsilon'], root)
    means, sigmas = env.regimes(cfg)
    dose = np.asarray(means[1:5]) - means[0]
    np.testing.assert_array_equal(dose, np.eye(4) * .035)
    np.testing.assert_array_equal(sigmas[1:5], np.broadcast_to(sigmas[0], (4, 4)))
    rng = np.random.default_rng(seed)
    observed = {}
    audited = {}
    for name, keep, factors in (('fit', fit, (1., np.sqrt(2.))), ('check', check, (-.5,))):
        baseline = center[keep] + means[0] + noise * rng.standard_normal(center[keep].shape)
        displacements = (np.asarray(factors)[:, None, None] * dose[None]).reshape(-1, 4)
        counter = baseline[None] + displacements[:, None, None]
        if not np.allclose(counter - baseline[None], displacements[:, None, None], rtol=0, atol=1e-14):
            raise ValueError('Physical translation did not match its design')
        observed[name] = dict(
            baseline=np.ascontiguousarray(transform(baseline, mix), dtype=np.float32),
            branches=np.ascontiguousarray(transform(counter, mix), dtype=np.float32),
            shifts=displacements,
        )
        audited[name] = {key: owner.identity(values) for key, values in observed[name].items()}
    return observed, dict(seed=seed, contexts=list(times), source_replay_maxabs=error, hashes=audited)
