"""One-sided certificates for linear objectives over anchored slope graphs."""

import math
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack


class LinearCertificate:
    """Solve y[right] - y[left] <= capacities with y[anchor] fixed to zero.

    The slope range [.7, 1.3] supplies a known box containing every physical
    candidate. Primal solutions are proposals. Only the one-sided objective
    bounds are certificates, and they do not depend on solver optimality.
    """

    def __init__(self, nodes, left_indices, right_indices, capacities, anchor_index):
        self.nodes = np.asarray(nodes, dtype=float)
        self.left = np.asarray(left_indices, dtype=np.intp)
        self.right = np.asarray(right_indices, dtype=np.intp)
        self.capacities = np.asarray(capacities, dtype=float)
        self.anchor = int(anchor_index)
        n, m = self.nodes.size, self.capacities.size
        if (self.nodes.shape != (n,) or self.capacities.shape != (m,)
                or self.left.shape != (m,) or self.right.shape != (m,)
                or not 0 <= self.anchor < n
                or np.any((self.left < 0) | (self.left >= n))
                or np.any((self.right < 0) | (self.right >= n))
                or not np.isfinite(self.nodes).all() or not np.isfinite(self.capacities).all()):
            raise ValueError('invalid slope graph')

        dx = self.nodes - self.nodes[self.anchor]
        dx_lo = np.nextafter(dx, -np.inf)
        dx_hi = np.nextafter(dx, np.inf)
        slope_lo = np.nextafter(.7, -np.inf)
        slope_hi = np.nextafter(1.3, np.inf)
        self.lower = np.nextafter(np.minimum(dx_lo * slope_lo, dx_lo * slope_hi), -np.inf)
        self.upper = np.nextafter(np.maximum(dx_hi * slope_lo, dx_hi * slope_hi), np.inf)
        self.lower[self.anchor] = self.upper[self.anchor] = 0.
        if not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all():
            raise ValueError('nonfinite variable box')
        self.radius = np.maximum(abs(self.lower), abs(self.upper))
        self.bounds = np.column_stack((self.lower, self.upper))
        rows = np.arange(m)
        self.matrix = coo_matrix((np.tile((-1., 1.), (m, 1)).ravel(),
                                  (np.repeat(rows, 2), np.column_stack((self.left, self.right)).ravel())),
                                 shape=(m, n)).tocsr()
        # The factor covers all operations in a bin plus cancellation against c.
        self.residual_roundoff = 4 * (m + 8) * np.finfo(float).eps
        self.underflow_slack = (m + 8) * np.finfo(float).tiny
        self._center_ready = False
        self._center = None

    def _solve(self, c, matrix, bounds):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected:.*',
                                    category=OptimizeWarning)
            return linprog(c, A_ub=matrix, b_ub=self.capacities, bounds=bounds,
                           method='highs-ds', options={'threads': 1,
                                                      'primal_feasibility_tolerance': 1e-9,
                                                      'dual_feasibility_tolerance': 1e-9})

    def _lower_bound(self, c, multiplier):
        # For lambda >= 0, c'y >= -lambda'b + (c + A'lambda)'y on Ay <= b.
        # Minimize the corrected linear term over the known coordinate box.
        n = self.nodes.size
        incoming = np.bincount(self.right, multiplier, minlength=n)
        outgoing = np.bincount(self.left, multiplier, minlength=n)
        residual = c + incoming - outgoing
        residual_error = self.residual_roundoff * (abs(c) + incoming + outgoing)
        residual_error = np.nextafter(residual_error + self.underflow_slack, np.inf)
        box_min = np.nextafter(np.minimum(residual * self.lower, residual * self.upper), -np.inf)
        correction = np.nextafter(residual_error * self.radius, np.inf)
        penalty = np.nextafter(multiplier * self.capacities, np.inf)
        terms = np.concatenate((box_min, -correction, -penalty))
        if not np.isfinite(terms).all():
            return -math.inf
        try:
            total = math.fsum(terms)
        except OverflowError:
            return -math.inf
        # A second step also covers a final double rounding in fsum.
        return math.nextafter(math.nextafter(total, -math.inf), -math.inf)

    def minimize(self, c):
        """Return (certified lower bound, numerical primal proposal or None)."""
        c = np.asarray(c, dtype=float)
        if c.shape != self.nodes.shape or not np.isfinite(c).all():
            raise ValueError('invalid linear objective')
        result = self._solve(c, self.matrix, self.bounds)
        multiplier = np.zeros_like(self.capacities)
        if result.success:
            candidate = -np.asarray(result.ineqlin.marginals)
            if np.isfinite(candidate).all():
                multiplier = np.maximum(candidate, 0.)
        proposal = result.x if result.success and np.isfinite(result.x).all() else None
        return self._lower_bound(c, multiplier), proposal

    def maximize(self, c):
        """Return (certified upper bound, numerical primal proposal or None)."""
        lower, proposal = self.minimize(-np.asarray(c, dtype=float))
        return -lower, proposal

    def is_feasible(self, proposal):
        """Check a proposal against the exact binary-float graph and anchor."""
        if proposal is None:
            return False
        y = np.asarray(proposal, dtype=float)
        if (y.shape != self.nodes.shape or not np.isfinite(y).all() or y[self.anchor] != 0.
                or np.any(y < self.lower) or np.any(y > self.upper)):
            return False
        return all(math.fsum((y[j], -y[i], -cap)) <= 0.
                   for i, j, cap in zip(self.left, self.right, self.capacities, strict=True))

    def feasible_center(self):
        """Return a max-margin primal proposal, allowing zero margin on exact faces."""
        if not self._center_ready:
            n = self.nodes.size
            distance = abs(self.nodes[self.right] - self.nodes[self.left])
            scale = np.minimum(1., distance)
            matrix = hstack((self.matrix, csr_matrix(scale[:, None])), format='csr')
            bounds = np.vstack((self.bounds, (0., 1.)))
            objective = np.r_[np.zeros(n), -1.]
            result = self._solve(objective, matrix, bounds)
            if result.success and np.isfinite(result.x).all():
                self._center = result.x[:-1]
            self._center_ready = True
        return None if self._center is None else self._center.copy()
