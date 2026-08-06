"""Runtime hardening: must be the FIRST import in every script.

Setting thread-count env vars BEFORE numpy / torch / sklearn are imported
prevents an intermittent Windows segfault that surfaces when several
subprocess invocations of these libraries happen back-to-back.

Add the line
    import _runtime_guard  # noqa: F401  (must be first)
as the very first non-docstring import in every entry-point script.
"""

import os

for _key in ("OMP_NUM_THREADS",
             "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")

# Deterministic hash seed for Python 3.3+ (otherwise PYTHONHASHSEED is random
# across processes and breaks cache-style deduplication of CSVs / pickles).
os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    import numpy as _np
    _np.random.seed(0)
except Exception:                                   # pragma: no cover
    pass