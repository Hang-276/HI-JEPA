"""Integrate reference responses supplied at explicitly queried anchor nodes.

Only visible paths and the supplied anchor Jacobians enter this module. Nearest
anchor reuse is an approximation, not a bound on unqueried response error. The
unchanged dense graph solver's consistency checks remain falsification checks.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


_GRAPH_PATH = Path(__file__).resolve().parents[1] / 'fine_reference_graph_20261111' / 'reference_graph.py'
_SPEC = importlib.util.spec_from_file_location('frozen_reference_graph', _GRAPH_PATH)
_GRAPH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GRAPH)


def reconstruct_from_anchors(z, anchor_jacobian, anchor_ids):
    """Return (reference sd or None, JSON-serializable diagnostics).

    ``z`` is (N,T,4); ``anchor_jacobian`` is (m,4,4), with the first matrix
    axis naming physical perturbations; ``anchor_ids`` names m unique flattened
    path nodes in the same order as those matrices. Distances use the visible
    standard deviations over all N*T nodes. Queried nodes retain their supplied
    responses even when visible coordinates tie. If every node is queried, a
    direct reorder bypasses nearest-neighbor interpolation entirely.
    """
    z = np.asarray(z, dtype=np.float64)
    responses = np.asarray(anchor_jacobian, dtype=np.float64)
    ids = np.asarray(anchor_ids)
    if (z.ndim != 3 or z.shape[-1] != 4 or z.shape[0] * z.shape[1] < 2
            or not np.isfinite(z).all()):
        raise ValueError('Expected finite reference paths with shape (N,T,4)')
    count = z.shape[0] * z.shape[1]
    if (ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= count)
            or np.unique(ids).size != ids.size):
        raise ValueError('Anchor IDs must be unique integer flat-node indices')
    ids = ids.astype(np.int64, copy=False)
    if responses.shape != (ids.size, 4, 4) or not np.isfinite(responses).all():
        raise ValueError('Expected finite anchor Jacobians with shape (m,4,4)')

    metadata = dict(queried_count=int(ids.size), nodecount=int(count),
                    anchor_ids_sha256=hashlib.sha256(ids.astype('<i8', copy=False).tobytes()).hexdigest(),
                    response_assignment='direct_all_nodes' if ids.size == count else 'nearest_queried_anchor',
                    queried_nodes_retain_own_response=True)
    if ids.size == count:
        assigned = np.empty((count, 4, 4), dtype=np.float64)
        assigned[ids] = responses
        metadata.update(max_nearest_standardized_distance=0., mean_nearest_standardized_distance=0.)
    else:
        points = z.reshape(-1, 4)
        visible_scale = points.std(axis=0)
        if np.any(visible_scale <= 0) or not np.isfinite(visible_scale).all():
            metadata.update(available=False, reason='zero_visible_reference_scale',
                            points=int(count), max_nearest_standardized_distance=None,
                            mean_nearest_standardized_distance=None)
            return None, metadata
        standardized = points / visible_scale
        distances, nearest = cKDTree(standardized[ids]).query(standardized, k=1, workers=1)
        assigned = responses[nearest].copy()
        assigned[ids] = responses
        distances[ids] = 0.
        metadata.update(max_nearest_standardized_distance=float(distances.max()),
                        mean_nearest_standardized_distance=float(distances.mean()))
    scale, diagnostics = _GRAPH.reconstruct_reference(z, assigned.reshape(z.shape + (4,)))
    diagnostics.update(metadata)
    return scale, diagnostics


def self_check():
    """Deterministic engineering checks; no Fine data or scientific reruns."""
    index = np.arange(95, dtype=float).reshape(5, 19)
    physical = np.stack((np.sin(index * .13), np.cos(index * .23),
                         .1 * index + np.sin(index * .31), np.cos(index * .37)), axis=-1)
    matrix = np.array([[1.2, .2, -.1, .1], [.1, .8, .2, 0.],
                       [-.2, .1, 1.1, .1], [.1, -.1, .05, .9]])
    visible = physical @ matrix + np.array([1., -2., .3, -.7])
    response = np.broadcast_to(matrix, visible.shape + (4,))
    direct, direct_diag = _GRAPH.reconstruct_reference(visible, response)
    ids = np.arange(index.size, dtype=np.int64)[::-1]
    varying = response.copy().reshape(-1, 4, 4)
    varying[:, 0, 0] += 1e-4 * np.sin(np.arange(index.size))
    varying_direct, varying_diag = _GRAPH.reconstruct_reference(visible, varying.reshape(response.shape))
    full, full_diag = reconstruct_from_anchors(visible, varying[ids], ids)
    sparse, sparse_diag = reconstruct_from_anchors(visible, matrix[None], np.array([17], dtype=np.int64))
    assert direct is not None and varying_direct is not None and full is not None and sparse is not None
    assert np.array_equal(full, varying_direct) and np.array_equal(sparse, direct)
    assert full_diag['solver_status'] == varying_diag['solver_status']
    assert full_diag['queried_count'] == index.size and sparse_diag['queried_count'] == 1
    assert full_diag['max_nearest_standardized_distance'] == 0.
    assert sparse_diag['max_nearest_standardized_distance'] > 0.
    error = float(np.max(np.abs(sparse - physical.std(axis=(0, 1)))))
    assert error < 1e-8
    for invalid in (np.array([0, 0]), np.array([-1]), np.array([index.size]), np.array([0.5])):
        try:
            reconstruct_from_anchors(visible, np.repeat(matrix[None], len(invalid), axis=0), invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid anchor IDs accepted')
    return dict(full_budget_exact_dense_equivalence=True, one_anchor_exact_affine_equivalence=True,
                affine_sd_maximum_error=error, invalid_ids_rejected=True,
                relative_edge_rms=direct_diag['relative_edge_rms'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as payload:
        scale, diagnostics = reconstruct_from_anchors(payload['z'], payload['anchor_jacobian'], payload['anchor_ids'])
    result = dict(scale=None if scale is None else scale.tolist(), diagnostics=diagnostics)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
