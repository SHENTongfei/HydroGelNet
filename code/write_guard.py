"""write_guard.py -- purge-before-write helpers for ALL artifact types.

The WorkBuddy host holds a permanent handle on files that already existed
when the session started. Any later write to such a file raises
PermissionError 13. The fix, proven 3x, is to DELETE the target first so the
write becomes a fresh-file creation that the host never locks.

Use these helpers instead of raw open()/to_csv()/json.dump() for every
artifact produced by a long compute phase. They implement the same L1/L2/L3
layers as _runtime_guard but for arbitrary writers.

Usage:
    import write_guard as wg
    wg.write_text(path, text)
    wg.write_json(path, obj)
    wg.write_bytes(path, data)
    wg.writer(path)          # context manager yielding an open file handle
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterator, Optional

RETRIES = 30
BACKOFF = 4.0


def safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except PermissionError:
        pass
    except Exception:
        pass


def _log_failure(canonical: str, retry: Optional[str], reason: str) -> None:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(root, "results", "arch_eval_logs")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "write_failures.log"), "a",
                  encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                     f"{canonical} -> {retry if retry else 'LOST'} | "
                     f"{reason}\n")
    except Exception:
        pass


def _open_with_guard(path: str, mode: str, *a, **kw):
    """Open path with purge-before-write + retry + .retry fallback."""
    safe_remove(path)
    try:
        return open(path, mode, *a, **kw)
    except PermissionError:
        pass
    for _i in range(RETRIES):
        time.sleep(BACKOFF)
        safe_remove(path)
        try:
            return open(path, mode, *a, **kw)
        except PermissionError:
            continue
        except Exception:
            raise
    # L3 fallback: same dir, .retry suffix
    retry_path = f"{path}.retry"
    safe_remove(retry_path)
    fh = open(retry_path, mode, *a, **kw)
    _log_failure(path, retry_path, f"PermissionError after {RETRIES * BACKOFF}s")
    return fh


def write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    with _open_with_guard(path, "w", encoding=encoding) as fh:
        fh.write(text)


def write_json(path: str, obj: Any, **kw) -> None:
    text = json.dumps(obj, ensure_ascii=False, **kw)
    write_text(path, text)


def write_bytes(path: str, data: bytes) -> None:
    with _open_with_guard(path, "wb") as fh:
        fh.write(data)


def writer(path: str, mode: str = "w", *a, **kw) -> Iterator:
    fh = _open_with_guard(path, mode, *a, **kw)
    try:
        yield fh
    finally:
        fh.close()
