"""Run the fixed sparse graph solver without importing the torch environment."""

import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import scipy
from reference_graph import reconstruct_reference, self_check


if __name__ == '__main__':
    if sys.argv[1] == 'self_check':
        result = dict(check=self_check(), runtime=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__))
        with Path(sys.argv[2]).open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    else:
        with np.load(sys.argv[1], allow_pickle=False) as data:
            sd, diagnostic = reconstruct_reference(data['z'], data['jacobian'])
        diagnostic['runtime'] = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__)
        with Path(sys.argv[2]).open('x') as stream:
            json.dump(dict(sd=None if sd is None else sd.tolist(), diagnostics=diagnostic), stream,
                      indent=2, allow_nan=False)
            stream.write('\n')
