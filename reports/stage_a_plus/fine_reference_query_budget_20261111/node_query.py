"""Restricted named-dose queries at requested visible reference nodes only."""

import copy
import importlib.util
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parent.parent / 'fine_local_dose_window_20261110' / 'query_data.py'
_spec = importlib.util.spec_from_file_location('frozen_node_query_adapter', _path)
_query = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_query)


def node_branches(public, receipt, flat_node_ids, dose):
    """Return only (requested node, physical dose axis, public observation axis)."""
    prior, shift = _query.prior, _query.shift
    slot, epsilon = prior.SLOT, prior.TARGET_EPS
    root = shift._stream_seed(slot, receipt['seed'], 'evaluation')
    tag, count = shift.FRESH_TAG, receipt['count']
    owner, env, transform, config, _ = shift._recipe(slot, epsilon, root)
    require = shift.base._require
    require((receipt['slot'], receipt['epsilon'], receipt['trajectory_seed'],
             receipt['trajectory_tag'], receipt['role'], receipt['namespace']) ==
            (slot, epsilon, root, tag, 'evaluation', shift.FRESH_NAMESPACE), 'Unexpected node receipt')

    def guard(cfg, t, n, capability):
        require(cfg == config and t == tag and n == count and capability is None,
                'Unexpected restricted node replay')

    namespace = {'np': np, 'copy': copy, 'env': env, 'require': require,
                 'ah': owner.array_sha, '_draw_guard': guard}
    owner._extract(owner.FHN_SOURCE, ('_draw', '_physical'), namespace)
    physical, actions, regimes, streams = namespace['_physical'](config, tag, count)
    mix = {'data': {'dimension': 4, 'data_seed': receipt['observation_mix_seed'],
                    'observation_layers': 4, 'observation_strength': .25}}

    def observe(state):
        return np.ascontiguousarray(transform(state, mix), dtype=np.float32)

    reproduced = dict(x=observe(physical), actions=actions, regimes=regimes)
    require(shift.base._identities(reproduced) == shift.base._identities(public) == receipt['public_arrays'],
            'Node-query public replay mismatch')
    require(all(streams[key] == value for key, value in receipt['physical_rng_stream_hashes'].items()),
            'Node-query innovation replay mismatch')
    ids = np.asarray(flat_node_ids, dtype=np.int64)
    require(ids.ndim == 1 and len(ids) > 0 and len(np.unique(ids)) == len(ids)
            and np.all((ids >= 0) & (ids < physical.shape[0] * physical.shape[1])),
            'Invalid node query IDs')
    support = physical.reshape(-1, 4)[ids, None, :]
    offsets = float(dose) * np.eye(4)
    plus, minus = observe(support + offsets), observe(support - offsets)
    return plus, minus, dict(public_parity=True, query_nodes=len(ids), physical_branches=8 * len(ids),
                            query_shape=list(plus.shape), node_ids_sha256=owner.array_sha(ids),
                            branch_sha256=dict(plus=owner.array_sha(plus), minus=owner.array_sha(minus)))
