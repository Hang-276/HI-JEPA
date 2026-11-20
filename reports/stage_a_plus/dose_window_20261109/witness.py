"""Post-result ambiguity witnesses using only previously sealed observations."""
import hashlib
import math

import run as model
import numpy as np


def witness_levels(x, weights, ell):
    phase = np.remainder(x / ell, 1.)
    assert np.all((phase > 0) & (phase < 1))
    order = np.argsort(phase, axis=1, kind='stable')
    u = np.take_along_axis(phase, order, axis=1)
    b = np.take_along_axis(weights, order, axis=1)
    tails = np.column_stack([np.cumsum(b[:, ::-1], axis=1)[:, ::-1], np.zeros(len(b))])
    valid = np.column_stack([np.ones(len(u), dtype=bool), u[:, 1:] > u[:, :-1],
                             np.ones(len(u), dtype=bool)])
    thresholds = np.column_stack([u, np.ones(len(u))])
    idx_lo = np.argmin(np.where(valid, tails, np.inf), axis=1)
    idx_hi = np.argmax(np.where(valid, tails, -np.inf), axis=1)
    row = np.arange(len(u))
    return phase, thresholds[row, idx_lo], thresholds[row, idx_hi]


def witnessed_ambiguous(x, d, cutoff, weights, ell):
    phase, lower, upper = witness_levels(x, weights, ell)
    q = max(math.exp(-2 * np.pi ** 2 / ell ** 2), np.finfo(float).tiny)
    a = 2 * q / (1 - q)
    f = 2 * a
    variance_min = 1 - f + ell ** 2 * (1 - a) / 12
    variance_max = 1 + f + ell ** 2 * (1 + a) / 12
    below, above = d.copy(), d.copy()
    for threshold in (lower, upper):
        p = (phase >= threshold[:, None]).astype(float) - phase
        witness = x + ell * p
        numerator = model.scores(witness[..., None])[:, 0]
        below = np.minimum(below, numerator / variance_min + model.EPS)
        above = np.maximum(above, numerator / variance_max - model.EPS)
    return (below < cutoff) & (above > cutoff)


def main():
    lock = model.read('LOCK.json')
    assert all(model.digest(k) == v for k, v in lock['sources'].items())
    receipt = model.read('FORECAST.json')
    assert receipt['lock'] == model.digest('LOCK.json')
    assert receipt['predictions'] == model.digest('FORECAST.npz')
    runs = {}
    with np.load(model.HERE / 'FORECAST.npz', allow_pickle=False) as data:
        for length, (groups, n) in model.LENGTHS.items():
            counts = np.zeros((len(model.ELLS), 2), dtype=np.int64)
            for group in range(groups):
                key = f'{length}_{group}'
                x = model.trajectory(model.FORMAL, length, group, n)
                assert hashlib.sha256(x.tobytes()).hexdigest() == receipt['public'][key]
                d, cutoff = data[key + '_d'].T
                w = model.cross_weights(x[:, :, 0])
                flags = data[key + '_f']
                for i, ell in enumerate(model.ELLS):
                    inner = witnessed_ambiguous(x[:, :, 0], d, cutoff, w, ell)
                    outer = ~flags[i, :, 0]
                    assert np.all(~inner | outer)
                    counts[i] += [inner.sum(), outer.sum()]
            runs[str(length)] = [dict(ell=float(ell), n=groups * n, witnessed_ambiguous=int(inner),
                                      outer_uncertified=int(outer), unresolved_difference=int(outer - inner))
                                 for ell, (inner, outer) in zip(model.ELLS, counts, strict=True)]
    model.save('WITNESS_AUDIT.json', dict(utc=model.utc(), forecast=model.digest('FORECAST.json'),
                                         script=model.digest('witness.py'), runs=runs))


if __name__ == '__main__':
    main()
