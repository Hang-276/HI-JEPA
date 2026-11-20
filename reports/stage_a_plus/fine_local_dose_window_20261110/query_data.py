"""Simulator-side endpoint branches. Only observables leave this adapter."""

import copy
import importlib.util
from pathlib import Path

import numpy as np

STAGE3 = Path(__file__).resolve().parent.parent / 'fine_stage3_20261110'
spec = importlib.util.spec_from_file_location('prior_stage3_data', STAGE3 / 'data_access.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
shift = prior.ac.src.shift_data


def public_panel(count, seed):
    return shift.fresh_observations(prior.SLOT, prior.TARGET_EPS, count, seed, 'evaluation')


def endpoint_branches(public, receipt, selected, dose):
    """Return two [window,time,dose-axis,observation-axis] arrays and an audit."""
    slot, epsilon = prior.SLOT, prior.TARGET_EPS
    root = shift._stream_seed(slot, receipt['seed'], 'evaluation')
    tag, count = shift.FRESH_TAG, receipt['count']
    owner, env, transform, config, _ = shift._recipe(slot, epsilon, root)
    require = shift.base._require
    require((receipt['slot'], receipt['epsilon'], receipt['trajectory_seed'],
             receipt['trajectory_tag'], receipt['role'], receipt['namespace']) ==
            (slot, epsilon, root, tag, 'evaluation', shift.FRESH_NAMESPACE),
            'Unexpected branch receipt')

    def guard(cfg, t, n, capability):
        require(cfg == config and t == tag and n == count and capability is None,
                'Unexpected conditional branch replay')

    namespace = {'np': np, 'copy': copy, 'env': env, 'require': require,
                 'ah': owner.array_sha, '_draw_guard': guard}
    owner._extract(owner.FHN_SOURCE, ('_draw', '_physical'), namespace)
    physical, actions, regimes, streams = namespace['_physical'](config, tag, count)
    mix = {'data': {'dimension': 4, 'data_seed': receipt['observation_mix_seed'],
                    'observation_layers': 4, 'observation_strength': .25}}

    def observe(state):
        return np.ascontiguousarray(transform(state, mix), dtype=np.float32)

    reproduced = dict(x=observe(physical), actions=actions, regimes=regimes)
    require(shift.base._identities(reproduced) == shift.base._identities(public)
            == receipt['public_arrays'], 'Public baseline replay mismatch')
    require(all(streams[key] == value for key, value in receipt['physical_rng_stream_hashes'].items()),
            'Physical innovation replay mismatch')
    support = physical[selected, :, None, :]
    offsets = dose * np.eye(4)
    plus, minus = observe(support + offsets), observe(support - offsets)
    return plus, minus, dict(public_parity=True, query_shape=list(plus.shape),
                            branch_sha256=dict(plus=owner.array_sha(plus), minus=owner.array_sha(minus)))
