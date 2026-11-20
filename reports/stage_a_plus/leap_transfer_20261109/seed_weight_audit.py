"""Apply the frozen equal-seed convention to each seed's warning rates."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ('native', 'magnitude', 'signed')


def summarize(records, views, weights):
    rates, uncertainty = {}, {}
    for metric in (f'{m}_{k}' for m in METHODS for k in ('hit', 'fpr')):
        per_seed = np.array([[p[metric] for p in records[v]] for v in views], dtype=np.float64)
        counts = per_seed.sum(axis=1)
        rates[metric] = float((counts[:, 0] / counts[:, 1]).mean())
        sampled = np.einsum('bp,spd->sbd', weights, per_seed)
        uncertainty[metric] = (sampled[..., 0] / sampled[..., 1]).mean(axis=0)
    contrasts = {}
    for comparator in ('native', 'magnitude'):
        distribution = uncertainty['signed_hit'] - uncertainty[f'{comparator}_hit']
        contrasts[comparator] = dict(gain=rates['signed_hit'] - rates[f'{comparator}_hit'],
                                     lower=float(np.quantile(distribution, .05 / 3)),
                                     upper=float(np.quantile(distribution, .975)))
    fpr = {m: float(np.quantile(uncertainty[f'{m}_fpr'], 1 - .05 / 3)) for m in METHODS}
    return dict(rates=rates, contrasts=contrasts, false_alarm_upper=fpr,
                warning_gate=all(c['gain'] >= .10 and c['lower'] > 0 for c in contrasts.values())
                and max(fpr.values()) < .05)


def main():
    path = HERE / 'SUMMARY.json'
    blob = path.read_bytes()
    records = json.loads(blob)['panel_statistics']
    n = len(records['leap0'])
    rng = np.random.default_rng(2026266100)
    weights = np.stack([np.bincount(row, minlength=n) for row in rng.integers(n, size=(4096, n))])
    result = dict(summary_sha=hashlib.sha256(blob).hexdigest())
    for family in ('leap', 'fine'):
        result[family] = summarize(records, tuple(f'{family}{i}' for i in range(3)), weights)
    with (HERE / 'EQUAL_SEED_AUDIT.json').open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
