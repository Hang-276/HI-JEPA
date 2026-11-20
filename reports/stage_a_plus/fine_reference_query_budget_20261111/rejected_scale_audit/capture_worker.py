"""Read an immutable graph solver's already-computed solution at return."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / 'sparse_worker.py'
_SPEC = importlib.util.spec_from_file_location('frozen_sparse_scale_audit', _PATH)
_SPARSE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SPARSE)


def capture(z, anchor_jacobian, anchor_ids):
    observed = {}
    code = _SPARSE._GRAPH.reconstruct_reference.__code__

    def trace(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == 'return' and 'solution' in frame.f_locals:
            solution = frame.f_locals['solution']
            observed['solution_scale'] = solution.std(axis=0).tolist()
            observed['solution_finite'] = bool(np.isfinite(solution).all())
            observed['solver_status'] = list(frame.f_locals['status'])
        return trace

    prior_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        scale, diagnostics = _SPARSE.reconstruct_from_anchors(z, anchor_jacobian, anchor_ids)
    finally:
        sys.settrace(prior_trace)
    if not observed or not observed['solution_finite'] or any(observed['solver_status']):
        raise RuntimeError('No finite converged solution available for this diagnostic')
    observed['returned_scale'] = None if scale is None else scale.tolist()
    observed['diagnostics'] = diagnostics
    observed['accepted_scale_exact_parity'] = None if scale is None else bool(
        np.array_equal(scale, np.asarray(observed['solution_scale'])))
    if scale is not None:
        assert observed['accepted_scale_exact_parity']
    return observed


def self_check():
    index = np.arange(95, dtype=float).reshape(5, 19)
    physical = np.stack((np.sin(index * .13), np.cos(index * .23),
                         .1 * index + np.sin(index * .31), np.cos(index * .37)), axis=-1)
    matrix = np.array([[1.2, .2, -.1, .1], [.1, .8, .2, 0.],
                       [-.2, .1, 1.1, .1], [.1, -.1, .05, .9]])
    visible = physical @ matrix + np.array([1., -2., .3, -.7])
    record = capture(visible, matrix[None], np.array([17], dtype=np.int64))
    assert record['accepted_scale_exact_parity']
    return dict(accepted_scale_exact_parity=True, solver_status=record['solver_status'],
                maximum_affine_sd_error=float(np.max(np.abs(
                    np.asarray(record['solution_scale']) - physical.std(axis=(0, 1))))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        result = self_check()
    else:
        with np.load(args.input, allow_pickle=False) as payload:
            result = capture(payload['z'], payload['anchor_jacobian'], payload['anchor_ids'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
