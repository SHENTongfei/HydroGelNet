"""Runtime hardening: must be the FIRST import in every script.

Setting thread-count env vars BEFORE numpy / torch / sklearn are imported
prevents an intermittent Windows segfault that surfaces when several
subprocess invocations of these libraries happen back-to-back.

Add the line
    import _runtime_guard  # noqa: F401  (must be first)
as the very first non-docstring import in every entry-point script.

--------------------------------------------------------------------------- #
HOST-FILE-LOCK GUARD (three independent layers, so a long compute phase can
NEVER lose its outputs to a locked file again):
  L1  purge-before-write  -- delete the target path (if it exists) right
      before writing. The WorkBuddy host holds a permanent handle ONLY on
      files that already existed when the session started; freshly created
      files are never locked. Deleting first turns every write into a
      fresh-file creation and sidesteps PermissionError 13 entirely.
  L2  long retry with backoff -- if L1 still hits PermissionError (e.g. the
      host grabbed the handle between our delete and our open), retry with
      30 x 4s = 120s of backoff instead of 15 x 8s.
  L3  .retry fallback file -- if all attempts against the canonical path
      fail, write to "<path>.retry" and append a marker to
      results/arch_eval_logs/write_failures.log so the operator can recover
      the artifact without re-running the compute phase.
--------------------------------------------------------------------------- #
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

try:
    import time as _t
    import pandas as _pd

    _orig_to_csv = _pd.DataFrame.to_csv

    def _safe_remove(path: str) -> None:
        """Delete the target so the next write is a fresh-file creation."""
        try:
            os.remove(path)
        except FileNotFoundError:
            pass                                    # already gone: perfect
        except PermissionError:
            pass                                    # held open: try write anyway
        except Exception:
            pass

    def _retry_to_csv(self, path_or_buf, *args, **kwargs):
        if not isinstance(path_or_buf, str):
            return _orig_to_csv(self, path_or_buf, *args, **kwargs)

        # ---- L1: purge-before-write ------------------------------------- #
        _safe_remove(path_or_buf)
        try:
            return _orig_to_csv(self, path_or_buf, *args, **kwargs)
        except PermissionError:
            pass

        # ---- L2: long retry with backoff (30 x 4s = 120s) ---------------- #
        for _i in range(30):
            _t.sleep(4)
            _safe_remove(path_or_buf)
            try:
                return _orig_to_csv(self, path_or_buf, *args, **kwargs)
            except PermissionError:
                continue
            except Exception:
                raise

        # ---- L3: .retry fallback file so the artifact is NOT lost -------- #
        retry_path = f"{path_or_buf}.retry"
        try:
            _safe_remove(retry_path)
            _orig_to_csv(self, retry_path, *args, **kwargs)
            _log_write_failure(path_or_buf, retry_path, "PermissionError after 120s")
            return None
        except Exception as _e:                      # pragma: no cover
            _log_write_failure(path_or_buf, None, repr(_e))
            raise

    def _log_write_failure(canonical, retry, reason):
        """Append one line so the operator knows an artifact moved."""
        try:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results", "arch_eval_logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "write_failures.log"), "a",
                      encoding="utf-8") as fh:
                fh.write(f"{_t.strftime('%Y-%m-%d %H:%M:%S')} | "
                         f"{canonical} -> {retry if retry else 'LOST'} | "
                         f"{reason}\n")
        except Exception:
            pass

    _pd.DataFrame.to_csv = _retry_to_csv
except Exception:                                   # pragma: no cover
    pass
