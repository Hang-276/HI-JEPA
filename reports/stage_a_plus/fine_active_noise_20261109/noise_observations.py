"""Query same-context shifts with shared or independent branch noise."""
import numpy as np

from active_observations import context, shift


def replicate_pairs(receipt, fit, check, times, seed):
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
    dose = np.asarray(mean[1:5]) - mean[0]
    np.testing.assert_array_equal(dose, np.eye(4) * .035)
    np.testing.assert_array_equal(sigmas[1:5], np.broadcast_to(sigmas[0], (4, 4)))
    rng = np.random.default_rng(seed)
    physical_center = center[fit] + mean[0]
    baseline = physical_center[None] + noise * rng.standard_normal((2,) + physical_center.shape)
    shifts = (np.array([1., np.sqrt(2.)])[:, None, None] * dose[None]).reshape(-1, 4)
    exact = baseline[None] + shifts[:, None, None, None]
    assert np.allclose(exact - baseline[None], shifts[:, None, None, None], rtol=0, atol=1e-14)
    noisy = (physical_center[None, None] + shifts[:, None, None, None]
             + noise * rng.standard_normal(exact.shape))
    validation_center = center[check] + mean[0] + noise * rng.standard_normal(center[check].shape)
    check_shifts = -.5 * dose
    validation = validation_center[None] + check_shifts[:, None, None]
    assert np.allclose(validation - validation_center[None], check_shifts[:, None, None], rtol=0, atol=1e-14)

    def observations(b, y, d):
        return dict(baseline=np.ascontiguousarray(transform(b, mix), dtype=np.float32),
                    branches=np.ascontiguousarray(transform(y, mix), dtype=np.float32), shifts=d)

    panels = {'exact': observations(baseline, exact, shifts),
              'noisy': observations(baseline, noisy, shifts),
              'check': observations(validation_center, validation, check_shifts)}
    hashes = {arm: {name: owner.identity(x) for name, x in row.items()} for arm, row in panels.items()}
    return panels, dict(seed=seed, contexts=list(times), replay_maxabs=replay, hashes=hashes)
