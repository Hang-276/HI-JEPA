"""Change the shift-to-noise ratio while holding the noise realization fixed."""
import numpy as np

from active_observations import context, shift

LEVELS = (1, 2, 4)


def dose_pairs(receipt, fit, check, times, seed):
    slot, role = receipt['slot'], receipt['role']
    root = shift._stream_seed(slot, receipt['seed'], role)
    assert role == 'calibration' and receipt['namespace'] == shift.FRESH_NAMESPACE
    assert receipt['trajectory_seed'] == root and receipt['trajectory_tag'] == shift.FRESH_TAG
    assert receipt['observation_mix_seed'] == shift.base._old_seed(slot, 'mix')
    assert not np.any(fit & check) and int(fit.sum()) == int(check.sum()) == 24
    _, _, center, noise, _, transform, mix, replay = context._exact_contexts(
        slot, receipt['epsilon'], receipt['count'], root, shift.FRESH_TAG, np.asarray(times),
        receipt['public_arrays'], receipt['physical_rng_stream_hashes'])
    owner, env, _, cfg, _ = shift._recipe(slot, receipt['epsilon'], root)
    mean, sigmas = env.regimes(cfg)
    dose = mean[1:5] - mean[0]
    np.testing.assert_array_equal(dose, np.eye(4) * .035)
    np.testing.assert_array_equal(sigmas[1:5], np.broadcast_to(sigmas[0], (4, 4)))
    rng = np.random.default_rng(seed)
    physical_center = center[fit] + mean[0]
    baseline = physical_center[None] + noise * rng.standard_normal((2,) + physical_center.shape)
    independent = physical_center[None, None] + noise * rng.standard_normal((8,) + baseline.shape)
    check_center = center[check] + mean[0] + noise * rng.standard_normal(center[check].shape)
    check_shifts = -.5 * dose
    b = np.ascontiguousarray(transform(baseline, mix), dtype=np.float32)
    check_pairs = dict(baseline=np.ascontiguousarray(transform(check_center, mix), dtype=np.float32),
                       branches=np.ascontiguousarray(transform(check_center[None] + check_shifts[:, None, None], mix),
                                                     dtype=np.float32),
                       shifts=check_shifts)
    pairs = {}
    for level in LEVELS:
        shifts = (np.array([1., np.sqrt(2.)])[:, None, None] * (level * dose[None])).reshape(-1, 4)
        delta = shifts[:, None, None, None]
        for kind, state in (('exact', baseline[None]), ('noisy', independent)):
            branch = state + delta
            assert np.allclose(branch - state, delta, rtol=0, atol=1e-14)
            pairs[f'{kind}{level}'] = dict(baseline=b,
                                          branches=np.ascontiguousarray(transform(branch, mix), dtype=np.float32),
                                          shifts=shifts)
    audit = {arm: {name: owner.identity(x) for name, x in row.items()}
             for arm, row in (pairs | {'check': check_pairs}).items()}
    return pairs, check_pairs, dict(seed=seed, replay_maxabs=replay, hashes=audit)
