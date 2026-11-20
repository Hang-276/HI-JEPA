"""Post-reveal diagnostic of the physical reference scale, without rescoring methods."""
from pathlib import Path
import numpy as np

from run import HERE, EPS, SOURCE_EPS, base, rep, shift_data


def quantify(scores):
    order = base.point_sets(scores)
    sorted_scores = np.sort(scores, axis=1)
    gap = sorted_scores[:, 2] - sorted_scores[:, 1]
    unique, counts = np.unique(order, axis=0, return_counts=True)
    return dict(n=len(scores), near_05=int((gap <= .05).sum()), near_02=int((gap <= .02).sum()),
                gap_quantiles=np.quantile(gap, [.05, .25, .5, .75, .95]),
                top2_counts={str(list(k)): int(v) for k, v in zip(unique.tolist(), counts)})


def report(slot, panels, eps):
    moments, sums, squares, observations = [], [], [], []
    for p in panels:
        receipt = p['by_epsilon'][str(eps)]['evaluation']
        z = shift_data.replay_physical(receipt).astype(np.float64)
        moments.append(np.diagonal(base.score_matrix(z), axis1=1, axis2=2))
        flat = z.reshape(-1, 4)
        sums.append(flat.sum(0))
        squares.append((flat ** 2).sum(0))
        observations.append(len(flat))
    moments, sums, squares, observations = map(np.asarray, (moments, sums, squares, observations))
    history_scale = np.asarray(base.historical_match(slot, 0)['train_sd_z'])
    changed = np.empty_like(moments)
    scales = []
    indexes = np.arange(len(panels))
    for fold in range(2):
        fit = indexes % 2 != fold
        mean = sums[fit].sum(0) / observations[fit].sum()
        var = squares[fit].sum(0) / observations[fit].sum() - mean ** 2
        assert (var > 0).all()
        changed[~fit] = moments[~fit] / var
        scales.append(np.sqrt(var))
    return dict(source_sd=history_scale, target_diagnostic_sd=scales,
                source_reference=quantify((moments / history_scale ** 2).reshape(-1, 4)),
                target_crossfit_reference=quantify(changed.reshape(-1, 4)))


def main():
    f = HERE / 'FORECAST.json'
    outcomes = HERE / 'SUMMARY.json'
    assert base.read(outcomes)['forecast_sha'] == base.sha(f)
    data = base.read(f)['slots']
    stats = {s: {str(e): report(s, data[s], e) for e in EPS} for s in SOURCE_EPS}
    rep.write('REFERENCE_DIAGNOSTIC.json', dict(utc=base.now(), forecast=base.sha(f),
        summary=base.sha(outcomes), script=base.sha(Path(__file__)), stats=stats,
        scope='post-reveal physical scale diagnosis; no methods rescored'))


if __name__ == '__main__':
    main()
