"""Decompose half-dose errors using only the saved public query receipts."""
import numpy as np

import run as study


def components(x, target):
    # Orthogonally project each output's half-dose prediction onto its known dose.
    weight = (target * target).sum(0)
    alpha = (x * target).sum(0) / weight
    residual = x - target * alpha
    total = np.linalg.norm(target)
    scale_error = np.linalg.norm(target * (alpha - 1.)) / total
    shape_error = np.linalg.norm(residual) / total
    np.testing.assert_allclose(
        np.linalg.norm(x - target) ** 2 / total ** 2,
        scale_error ** 2 + shape_error ** 2,
        rtol=0, atol=1e-12,
    )
    adjusted = np.linalg.norm(residual / alpha) / total if np.all(abs(alpha) > 1e-8) else None
    return dict(alpha=alpha, scale_error=scale_error, shape_error=shape_error, adjusted_error=adjusted)


def audit_slot(slot, frozen):
    _, views, actions, _ = study.src.reference.historical(slot)
    fold = np.arange(len(actions)) % 4
    panels = []
    for row in frozen['panels']:
        receipt = row['calibration']
        queries, heldout, rebuilt = study.dose_pairs(
            receipt, fold == 0, fold == 1, study.ac.TIMES, row['query']['seed'])
        assert rebuilt == row['query']
        values = {}
        for view in ('fine', 'raw'):
            encode = views[view][1]
            center = encode(queries['exact1']['baseline']).reshape(-1, 4).mean(0)
            check, target = study.ac.feature_difference(encode, heldout, center, 3)
            for arm in study.ARMS:
                name = f'{view}_{arm}'
                design, train_target = study.an.stacked_difference(encode, queries[arm], center)
                beta, condition = study.an.solve(design, train_target, cross=False)
                assert beta is not None and condition == row['audit'][name]['condition']
                prediction = check @ beta
                np.testing.assert_allclose(np.linalg.norm(prediction - target) / np.linalg.norm(target),
                                           row['audit'][name]['check'], rtol=0, atol=1e-12)
                values[name] = components(prediction, target)
        panels.append(values)
    methods = panels[0].keys()
    return dict(
        panels=panels,
        medians={m: {k: np.median([p[m][k] for p in panels], axis=0)
                     for k in ('alpha', 'scale_error', 'shape_error', 'adjusted_error')}
                 for m in methods},
    )


def main():
    folder = study.HERE
    path = folder / 'FORECAST.json'
    lock = study.base.read(folder / 'LOCK.json')
    frozen = study.base.read(path)
    assert lock['sources'] == study.sources()
    assert frozen['lock_sha'] == study.base.sha(folder / 'LOCK.json')
    slots = {s: audit_slot(s, frozen['slots'][s]) for s in study.SLOTS}
    study.rep.write('SCALE_AUDIT.json', dict(utc=study.base.now(), forecast_sha=study.base.sha(path), slots=slots))


if __name__ == '__main__':
    main()
