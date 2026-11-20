"""Recover a shared reference scale from visible local response matrices."""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial import cKDTree


NEIGHBORS = 16
MAX_CONDITION = 20.
MIN_SINGULAR = 1e-3
SOLVER_RTOL = 1e-9
SOLVER_MAXITER = 5000
MAX_RELATIVE_EDGE_RMS = .02


def reconstruct_reference(z, jacobian):
    """Return (physical-unit sd or None, observable diagnostics).

    z has shape (trajectories, time, 4). The response Jacobian has shape
    (trajectories, time, physical_axis, z_axis). Each edge maps its visible
    row increment through the mean of its two inverse Jacobians. A shared
    graph potential integrates these edge displacements up to translation.
    """
    z = np.asarray(z, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if (z.ndim != 3 or z.shape[-1] != 4 or z.shape[0] * z.shape[1] < 2
            or jacobian.shape != z.shape + (4,)
            or not np.isfinite(z).all() or not np.isfinite(jacobian).all()):
        raise ValueError('Expected finite reference paths and row-physical response matrices')
    points = z.reshape(-1, 4)
    responses = jacobian.reshape(-1, 4, 4)
    singular = np.linalg.svd(responses, compute_uv=False)
    minimum = float(singular[:, -1].min())
    condition = np.divide(singular[:, 0], singular[:, -1],
                          out=np.full(len(points), np.finfo(float).max), where=singular[:, -1] > 0)
    maximum_condition = float(condition.max())
    diagnostics = dict(available=False, reason=None, points=len(points), neighbors=NEIGHBORS,
                       minimum_singular=minimum, maximum_condition=maximum_condition,
                       solver_rtol=SOLVER_RTOL, solver_maxiter=SOLVER_MAXITER,
                       edge_residual_threshold=MAX_RELATIVE_EDGE_RMS)
    if minimum < MIN_SINGULAR or maximum_condition > MAX_CONDITION:
        diagnostics['reason'] = 'local_response_condition'
        return None, diagnostics
    scale = points.std(axis=0)
    if np.any(scale <= 0):
        diagnostics['reason'] = 'zero_visible_reference_scale'
        return None, diagnostics
    standardized = points / scale
    count = len(points)
    query_count = min(NEIGHBORS + 1, count)
    _, neighbors = cKDTree(standardized).query(standardized, k=query_count, workers=1)
    owners = np.broadcast_to(np.arange(count)[:, None], neighbors.shape)
    nonself = owners != neighbors
    keep = nonself & (np.cumsum(nonself, axis=1) <= NEIGHBORS)
    edges = np.column_stack((owners[keep], neighbors[keep]))
    time_indices = np.arange(count).reshape(z.shape[:2])
    consecutive = np.column_stack((time_indices[:, :-1].ravel(), time_indices[:, 1:].ravel()))
    edges = np.unique(np.sort(np.concatenate((edges, consecutive)), axis=1), axis=0)
    left, right = edges.T
    row = np.r_[left, right]
    column = np.r_[right, left]
    adjacency = coo_matrix((np.ones(len(row)), (row, column)), shape=(count, count)).tocsr()
    components = int(connected_components(adjacency, directed=False, return_labels=False))
    diagnostics.update(edges=len(edges), component_count=components)
    if components != 1:
        diagnostics['reason'] = 'disconnected_graph'
        return None, diagnostics
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(degree, format='csr') - adjacency
    inverse = np.linalg.inv(responses)
    visible_step = points[right] - points[left]
    midpoint_inverse = (inverse[left] + inverse[right]) / 2
    target = (visible_step[:, None, :] @ midpoint_inverse)[:, 0, :]
    diagnostics.update(
        minimum_degree=int(degree.min()), maximum_degree=int(degree.max()),
        maximum_edge_length_standardized=float(np.linalg.norm(standardized[right] - standardized[left], axis=1).max()),
        maximum_edge_length_visible=float(np.linalg.norm(visible_step, axis=1).max()),
        maximum_edge_target_length=float(np.linalg.norm(target, axis=1).max()))
    rhs = np.zeros_like(points)
    np.add.at(rhs, left, -target)
    np.add.at(rhs, right, target)
    rhs -= rhs.mean(axis=0)
    operator = LinearOperator((count, count), matvec=lambda values: laplacian @ values + values.mean(), dtype=np.float64)
    preconditioner = LinearOperator((count, count),
                                   matvec=lambda values: values / (degree + 1 / count), dtype=np.float64)
    solution = np.empty_like(points)
    iterations, status, residual = [], [], []
    centered = points - points.mean(axis=0)
    for axis in range(4):
        iteration_count = [0]

        def count_iteration(_):
            iteration_count[0] += 1

        values, info = cg(operator, rhs[:, axis], x0=centered[:, axis], rtol=SOLVER_RTOL,
                          atol=0., maxiter=SOLVER_MAXITER, M=preconditioner, callback=count_iteration)
        solution[:, axis] = values - values.mean()
        iterations.append(iteration_count[0])
        status.append(int(info))
        norm = float(np.linalg.norm(rhs[:, axis]))
        residual.append(float(np.linalg.norm(operator @ values - rhs[:, axis]) / max(norm, np.finfo(float).tiny)))
    diagnostics.update(iterations=iterations, solver_status=status, solver_relative_residual=residual)
    if any(status) or not np.isfinite(solution).all():
        diagnostics['reason'] = 'integration_solver_failure'
        return None, diagnostics
    edge_error = solution[right] - solution[left] - target
    edge_rms = float(np.sqrt(np.mean(edge_error**2)))
    target_rms = float(np.sqrt(np.mean(target**2)))
    relative = edge_rms / max(target_rms, np.finfo(float).tiny)
    diagnostics.update(edge_rms=edge_rms, target_edge_rms=target_rms,
                       relative_edge_rms=relative,
                       per_axis_edge_rms=np.sqrt(np.mean(edge_error**2, axis=0)).tolist())
    if relative > MAX_RELATIVE_EDGE_RMS:
        diagnostics['reason'] = 'inconsistent_local_displacements'
        return None, diagnostics
    physical_sd = solution.std(axis=0)
    if np.any(physical_sd <= 0) or not np.isfinite(physical_sd).all():
        diagnostics['reason'] = 'invalid_reconstructed_scale'
        return None, diagnostics
    diagnostics.update(available=True, reference_scale=physical_sd.tolist())
    return physical_sd, diagnostics


def self_check():
    """Check orientation and translation invariance on a fixed affine map."""
    index = np.arange(95, dtype=float).reshape(5, 19)
    physical = np.stack((np.sin(index * .13), np.cos(index * .23),
                         .1 * index + np.sin(index * .31), np.cos(index * .37)), axis=-1)
    matrix = np.array([[1.2, .2, -.1, .1], [.1, .8, .2, 0.],
                       [-.2, .1, 1.1, .1], [.1, -.1, .05, .9]])
    visible = physical @ matrix + np.array([1., -2., .3, -.7])
    response = np.broadcast_to(matrix, visible.shape + (4,))
    sd, audit = reconstruct_reference(visible, response)
    if sd is None:
        raise AssertionError(audit)
    error = float(np.max(np.abs(sd - physical.std(axis=(0, 1)))))
    assert error < 1e-8 and audit['relative_edge_rms'] < 1e-8
    return dict(affine_sd_maximum_error=error, relative_edge_rms=audit['relative_edge_rms'],
                iterations=audit['iterations'], component_count=audit['component_count'])
