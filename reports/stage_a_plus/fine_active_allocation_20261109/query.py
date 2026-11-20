"""Construct nested fixed-budget queries for the repetition allocation test."""
import numpy as np

from active_observations import context, shift


ARMS = ('pooled2', 'mean2', 'mean8', 'mean32', 'exact2', 'exact32')


def queries(receipt, fit, check, times, seed):
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
    means, sigmas = env.regimes(cfg)
    dose = means[1:5] - means[0]
    np.testing.assert_array_equal(dose, np.eye(4) * .035)
    np.testing.assert_array_equal(sigmas[1:5], np.broadcast_to(sigmas[0], (4, 4)))
    shifts = (np.array([1., np.sqrt(2.)])[:, None, None] * dose[None]).reshape(8, 4)
    rng = np.random.default_rng(seed)
    physical = center[fit].reshape(-1, 4) + means[0]
    order = rng.permutation(len(physical))
    physical = physical[order]
    base = physical[None] + noise * rng.standard_normal((32,) + physical.shape)
    independent = physical[None, None] + noise * rng.standard_normal((8,) + base.shape)

    def make(r, exact):
        count = 384 // r
        b = base[:r, :count]
        y = b[None] if exact else independent[:, :r, :count]
        difference = shifts[:, None, None]
        branch = y + difference
        assert np.allclose(branch - y, difference, rtol=0, atol=1e-14)
        return dict(baseline=np.ascontiguousarray(transform(b, mix), dtype=np.float32),
                    branches=np.ascontiguousarray(transform(branch, mix), dtype=np.float32), shifts=shifts)

    panels = {'mean2': make(2, False), 'mean8': make(8, False), 'mean32': make(32, False),
              'exact2': make(2, True), 'exact32': make(32, True)}
    panels['pooled2'] = panels['mean2']
    c = center[check] + means[0] + noise * rng.standard_normal(center[check].shape)
    check_dose = -.5 * dose
    held = dict(baseline=np.ascontiguousarray(transform(c, mix), dtype=np.float32),
                branches=np.ascontiguousarray(transform(c[None] + check_dose[:, None, None], mix), dtype=np.float32),
                shifts=check_dose)
    hashes = {arm: {name: owner.identity(x) for name, x in row.items()}
              for arm, row in (panels | {'check': held}).items()}
    return panels, held, dict(seed=seed, context_order=order.tolist(), replay_maxabs=replay, hashes=hashes)
