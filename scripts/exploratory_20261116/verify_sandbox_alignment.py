#!/usr/bin/env python3
"""Read-only operator checks and reproducibility reruns; standard library + NumPy."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from phase_core import path_sufficient_stats, gaussian_variance_coefficients, top2_gap
from probe1_harmonic_and_subspace import observe, window_grams, function_metrics, HADAMARD

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/exploratory_20261116'
TIER = 'exploratory_numerical_sandbox'


def without_runtime(value):
    if isinstance(value, dict):
        return {k: without_runtime(v) for k, v in value.items()
                if 'second' not in k and k not in ('condition_timings',)}
    if isinstance(value, list):
        return [without_runtime(v) for v in value]
    return value


def main():
    started = time.perf_counter()
    namespace = {'np': np}
    metric_path = ROOT / 'src/slowrepr/metrics.py'
    tree = ast.parse(metric_path.read_text())
    names = ('_variance', '_top2_and_gap', 'score_windows')
    selected = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(selected) == 3
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(metric_path), 'exec'), namespace)
    score = namespace['score_windows']
    # Extract only the archived observation function. No SciPy, torch, runner,
    # model, training or archived prediction module is imported or executed.
    observation_path = ROOT / 'reports/lejepa_slow_rank_20261112/training/run.py'
    original_tree = ast.parse(observation_path.read_text())
    observation_node = next(x for x in original_tree.body if isinstance(x, ast.FunctionDef) and x.name == 'observe')
    class NumpyObservation(ast.NodeTransformer):
        """Translate only the three tensor primitives used in this expression."""
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id='np', ctx=node.ctx), node) if node.id == 'torch' else node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'square':
                return ast.copy_location(ast.BinOp(left=node.func.value, op=ast.Pow(), right=ast.Constant(value=2)), node)
            for keyword in node.keywords:
                if keyword.arg == 'dim':
                    keyword.arg = 'axis'
            return node

    obs_namespace = {'np': np, 'HADAMARD': HADAMARD}
    numpy_tree = ast.fix_missing_locations(NumpyObservation().visit(ast.Module(body=[observation_node], type_ignores=[])))
    exec(compile(numpy_tree, str(observation_path), 'exec'), obs_namespace)
    rng = np.random.default_rng(2026111603)
    states = rng.normal(size=(19, 4))
    observation_error = float(np.max(np.abs(obs_namespace['observe'](states)-observe(states))))
    rho, n, length, lags = np.array([.95, .91, .89, .82]), 11, 65, (4, 16, 32)
    stream_seed = 123451
    stream = path_sufficient_stats(np.random.default_rng(stream_seed), rho, n, (length,), lags)[length]
    generator = np.random.default_rng(stream_seed)
    z, noise = np.empty((n, length, 4)), np.empty((n, length, 4))
    for t in range(length):
        innovation = generator.standard_normal((n, 4))
        z[:, t] = innovation if t == 0 else rho*z[:, t-1]+np.sqrt(1-rho*rho)*innovation
        noise[:, t] = generator.standard_normal((n, 4))
    ref_z, ref_e = rng.normal(size=(2, 4096, 4))
    maximum_score_error = 0.
    exact_rank_matches = True
    for v in (0., .005, .2):
        prediction = z+np.sqrt(v)*noise
        variance = (ref_z+np.sqrt(v)*ref_e).var(axis=0, ddof=0)
        actual = score(prediction, z, np.ones(4), variance, lags)
        compact = (stream[0]+np.sqrt(v)*stream[1]+v*stream[2])/variance
        maximum_score_error = max(maximum_score_error, float(np.max(np.abs(compact-actual['predicted_scores']))))
        top, gap = top2_gap(compact)
        exact_rank_matches &= bool(np.array_equal(top, actual['predicted_top2']))
        maximum_score_error = max(maximum_score_error, float(np.max(np.abs(gap-actual['gap']))))
    # Stable tie breaking and threshold equality are part of the contract.
    tied = np.array([[1., 1., 1., 1.], [2., 1., 1., 2.]])
    tie_matches = all(np.array_equal(a, b) for a, b in zip(top2_gap(tied), namespace['_top2_and_gap'](tied)))
    gram = window_grams(np.concatenate((z, noise), axis=2))
    map8 = rng.normal(size=(8, 4))
    y = np.concatenate((z, noise), axis=2)@map8
    direct = score(y, z, np.ones(4), np.ones(4), lags)['predicted_scores']
    gram_error = float(np.max(np.abs(np.einsum('ai,nab,bi->ni', map8, gram, map8)-direct)))
    # Independent dense trace identity verifies exact endpoint and cross-lag terms.
    dense_error = 0.
    matrix = np.zeros((length, length))
    for lag in lags:
        diff = np.eye(length)[lag:]-np.eye(length)[:-lag]
        matrix += diff.T@diff/(len(lags)*(length-lag))
    indices = np.arange(length)
    for r in (.5, .905, .99):
        coefficients = gaussian_variance_coefficients(r, length)
        for v in (0., .05, .2):
            cov = r**np.abs(indices[:, None]-indices[None, :])+v*np.eye(length)
            dense = float(2*np.trace(matrix@cov@matrix@cov))
            compact = float(coefficients@np.array([1., v, v*v]))
            dense_error = max(dense_error, abs(dense-compact))
    identity = function_metrics(states[:, :2], states)
    phase = json.loads((OUT/'probe2_summary.json').read_text())['phase']
    perfect_zero = all(c['empirical_physical_ref_sensitivity']['all']['errors'] == 0
                       for c in phase['cells'] if c['r2_population'] == 1.)
    checks = {'observation_max_absolute_error': observation_error,
              'observation_check_scope': 'archived expression AST translated torch.sin/stack/square into NumPy; no torch import; float64 check, not bitwise float32 training parity',
              'stream_scores_and_gaps_max_absolute_error': maximum_score_error,
              'stream_top2_exact_match': exact_rank_matches, 'stable_tie_exact_match': tie_matches,
              'gram_scores_max_absolute_error': gram_error,
              'analytic_variance_vs_dense_trace_max_absolute_error': dense_error,
              'function_metric_identity_sin_theta': identity['function_subspace_sin_theta_max'],
              'perfect_readout_matched_reference_all_100_cells_exact_zero': perfect_zero}
    assert observation_error == 0 and maximum_score_error < 1e-12 and gram_error < 1e-11
    assert exact_rank_matches and tie_matches and dense_error < 1e-12 and perfect_zero
    assert identity['function_subspace_sin_theta_max'] < 1e-7
    rerun_dir = Path(tempfile.mkdtemp(prefix='verified_reproduction_', dir=OUT))
    records = []
    for probe in ('probe1_harmonic_and_subspace', 'probe2_phase_diagram_alignment'):
        script = ROOT/'scripts/exploratory_20261116'/f'{probe}.py'
        help_start = time.perf_counter()
        help_run = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True, check=True)
        assert 'usage:' in help_run.stdout
        help_elapsed = time.perf_counter()-help_start
        begin = time.perf_counter()
        run = subprocess.run([sys.executable, str(script), '--output-dir', str(rerun_dir)],
                             capture_output=True, text=True, check=True)
        elapsed = time.perf_counter()-begin
        filename = 'probe1_summary.json' if probe.startswith('probe1') else 'probe2_summary.json'
        first = json.loads((OUT/filename).read_text())
        again = json.loads((rerun_dir/filename).read_text())
        equal = without_runtime(first) == without_runtime(again)
        assert equal and elapsed < 60
        records.append({'script': str(script.relative_to(ROOT)), 'wall_seconds_including_startup_and_write': elapsed,
                        'help_seconds': help_elapsed, 'numeric_payload_exactly_reproduced': equal,
                        'script_receipt': json.loads(run.stdout)})
    repo_begin = time.perf_counter()
    repository = subprocess.run(['python3', 'scripts/check_repository.py'], cwd=ROOT,
                                capture_output=True, text=True)
    original_environment_failure = None
    if repository.returncode and 'file_digest' in repository.stderr:
        original_environment_failure = {'returncode': repository.returncode, 'stderr': repository.stderr,
            'explanation': 'Default python3 is older than 3.11; unchanged checker requires hashlib.file_digest. Rerun same command with available NumPy interpreter directory first in PATH.'}
        environment = dict(os.environ)
        environment['PATH'] = str(Path(sys.executable).parent)+os.pathsep+environment.get('PATH', '')
        repository = subprocess.run(['python3', 'scripts/check_repository.py'], cwd=ROOT,
                                    capture_output=True, text=True, env=environment)
    repo_elapsed = time.perf_counter()-repo_begin
    repo_output = json.loads(repository.stdout)
    assert repository.returncode == 0
    result = {'evidence_tier': TIER, 'status': 'passed',
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'compatibility': 'Python 3.11 syntax parsed below; actual numerical execution used available 3.13.11'},
              'checks': checks, 'reproducibility': records,
              'repository_check': {'command': 'python3 scripts/check_repository.py', 'returncode': repository.returncode,
                                   'wall_seconds_including_initial_version_retry': repo_elapsed,
                                   'stdout': repo_output, 'stderr': repository.stderr,
                                   'initial_environment_failure': original_environment_failure,
                                   'successful_python3_directory': str(Path(sys.executable).parent)},
              'protected_source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (metric_path, observation_path)}}
    scripts = sorted((ROOT/'scripts/exploratory_20261116').glob('*.py'))
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 11))
    result['python311_syntax_files_passed'] = len(scripts)
    result['elapsed_seconds'] = time.perf_counter()-started
    (OUT/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'evidence_tier': TIER, 'status': 'passed', 'elapsed_seconds': result['elapsed_seconds'],
                      'repository_seconds': repo_elapsed, 'operator_checks': checks, 'rerun_seconds': [r['wall_seconds_including_startup_and_write'] for r in records]}))


if __name__ == '__main__':
    main()
