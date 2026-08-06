"""Download raw data and PROVE that every link works.

Design rules (enforced by the skill):
  * Every source must be free, public and require no registration / API key.
  * Every URL is verified with a real HTTP request BEFORE it is written into
    DATA_SOURCES.md. A link that is not verified must never be shown to the user.
  * All files land in absolute paths under RAW_DIR / EXTERNAL_DIR.

The agent fills in SOURCES for the concrete project. Everything else is generic.

Usage
-----
    python download_data.py --verify-only
    python download_data.py
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import hashlib
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import paths

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
TIMEOUT = 60


# --------------------------------------------------------------------------- #
# Source registry -- FILL THIS IN PER PROJECT
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """One downloadable artefact."""

    name: str                      # short id, used as log key
    url: str                       # direct download URL
    dest: str                      # absolute destination path
    role: str                      # "internal" | "external" | "annotation"
    license: str                   # e.g. "CC-BY 4.0"
    citation: str                  # paper / database citation
    landing_page: str = ""         # human-facing page (for the paper)
    note: str = ""                 # anything the reader should know
    optional: bool = False         # if True, failure does not abort
    sha256: Optional[str] = None   # filled after download
    n_bytes: int = 0
    status: str = "PENDING"
    http_code: int = 0


SOURCES: List[Source] = [
    # ------------------------------------------------------------------ #
    # HydroGelNet: sheng-hu/hydrogels (MIT) -- Nature 2025 super-adhesive
    # hydrogel dataset. Internal = df_180 (round-1 baseline, train region);
    # external = df_341 minus df_180 (161 SMBO-guided high-performance
    # formulas = time-extrapolation test region). df_316 used as an
    # intermediate-scale ablation source.
    # ------------------------------------------------------------------ #
    Source(
        name="hydrogel_df180",
        url="https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_180.csv",
        dest=os.path.join(paths.RAW_DIR, "df_180.csv"),
        role="internal",
        license="MIT",
        citation="Liao H, Hu S, Yang H, et al. Data-driven de novo design of super-adhesive hydrogels. Nature, 2025. doi:10.1038/s41586-025-09269-4.",
        landing_page="https://github.com/sheng-hu/hydrogels",
        note="Round-1 baseline: 180 formulations, 6 monomer molar fractions -> Glass (kPa)_max adhesion. Train region (low-performance).",
    ),
    Source(
        name="hydrogel_df341",
        url="https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_341.csv",
        dest=os.path.join(paths.RAW_DIR, "df_341.csv"),
        role="external",
        license="MIT",
        citation="Liao H, Hu S, Yang H, et al. Data-driven de novo design of super-adhesive hydrogels. Nature, 2025. doi:10.1038/s41586-025-09269-4.",
        landing_page="https://github.com/sheng-hu/hydrogels",
        note="Full dataset 341 formulas. External set = rows not in df_180 (161 SMBO-guided high-performance formulas).",
    ),
    Source(
        name="hydrogel_df316",
        url="https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_316.csv",
        dest=os.path.join(paths.RAW_DIR, "df_316.csv"),
        role="annotation",
        license="MIT",
        citation="Liao H, Hu S, Yang H, et al. Data-driven de novo design of super-adhesive hydrogels. Nature, 2025. doi:10.1038/s41586-025-09269-4.",
        landing_page="https://github.com/sheng-hu/hydrogels",
        note="Intermediate round-3 dataset (316 formulas); used for data-size ablation.",
        optional=True,
    ),
]


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    handler = urllib.request.HTTPSHandler(context=ctx)
    op = urllib.request.build_opener(handler)
    op.addheaders = [("User-Agent", USER_AGENT)]
    return op


def verify_url(url: str, retries: int = 2) -> tuple[bool, int, str]:
    """Return (ok, http_code, message). Tries HEAD then a 1-byte ranged GET."""
    op = _opener()
    last_msg = ""
    for attempt in range(retries + 1):
        for method in ("HEAD", "GET"):
            try:
                req = urllib.request.Request(url, method=method)
                if method == "GET":
                    req.add_header("Range", "bytes=0-0")
                with op.open(req, timeout=TIMEOUT) as resp:
                    code = resp.getcode()
                    if code in (200, 206):
                        size = resp.headers.get("Content-Length", "?")
                        ctype = resp.headers.get("Content-Type", "?")
                        return True, code, f"{ctype}, Content-Length={size}"
                    last_msg = f"unexpected status {code}"
            except urllib.error.HTTPError as exc:
                last_msg = f"HTTPError {exc.code}"
                if exc.code in (403, 405) and method == "HEAD":
                    continue           # some servers reject HEAD, try GET
                if exc.code == 404:
                    return False, 404, "404 Not Found"
            except Exception as exc:                       # noqa: BLE001
                last_msg = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    return False, 0, last_msg


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(src: Source, force: bool = False) -> bool:
    """Stream a source to disk. Returns True on success."""
    if os.path.exists(src.dest) and not force:
        src.n_bytes = os.path.getsize(src.dest)
        src.sha256 = sha256_of(src.dest)
        src.status = "CACHED"
        print(f"  [cached] {src.name} -> {src.dest} ({src.n_bytes:,} bytes)")
        return True

    os.makedirs(os.path.dirname(src.dest), exist_ok=True)
    tmp = src.dest + ".part"
    op = _opener()
    try:
        req = urllib.request.Request(src.url)
        with op.open(req, timeout=TIMEOUT) as resp, open(tmp, "wb") as out:
            total = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                total += len(block)
                if total % (16 << 20) < (1 << 20):
                    print(f"    ... {total / 1e6:.1f} MB", flush=True)
        os.replace(tmp, src.dest)
        src.n_bytes = os.path.getsize(src.dest)
        src.sha256 = sha256_of(src.dest)
        src.status = "OK"
        print(f"  [ok]     {src.name} -> {src.dest} ({src.n_bytes:,} bytes)")
        return True
    except Exception as exc:                               # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        src.status = f"FAILED ({type(exc).__name__})"
        print(f"  [FAIL]   {src.name}: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_data_sources_md() -> str:
    """Emit DATA_SOURCES.md with verification evidence for every link."""
    lines: List[str] = []
    lines.append(f"# Data sources for {paths.MODEL_NAME}")
    lines.append("")
    lines.append(
        "Every link below was verified with a live HTTP request on "
        + time.strftime("%Y-%m-%d %H:%M:%S")
        + ". No registration, login, API key or data-access application is "
        "required for any of them."
    )
    lines.append("")
    lines.append("| # | Source | Role | HTTP | Size (bytes) | License | Link |")
    lines.append("|---|--------|------|------|--------------|---------|------|")
    for i, s in enumerate(SOURCES, 1):
        lines.append(
            f"| {i} | {s.name} | {s.role} | {s.http_code} {s.status} | "
            f"{s.n_bytes:,} | {s.license} | <{s.url}> |"
        )
    lines.append("")
    lines.append("## Details")
    for s in SOURCES:
        lines.append("")
        lines.append(f"### {s.name}")
        lines.append(f"- **Role**: {s.role}")
        lines.append(f"- **Direct download**: <{s.url}>")
        if s.landing_page:
            lines.append(f"- **Landing page**: <{s.landing_page}>")
        lines.append(f"- **Local absolute path**: `{s.dest}`")
        lines.append(f"- **License**: {s.license}")
        lines.append(f"- **Citation**: {s.citation}")
        if s.sha256:
            lines.append(f"- **SHA-256**: `{s.sha256}`")
        if s.note:
            lines.append(f"- **Note**: {s.note}")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(paths.DATA_SOURCES_MD), exist_ok=True)
    with open(paths.DATA_SOURCES_MD, "w", encoding="utf-8") as fh:
        fh.write(text)
    return paths.DATA_SOURCES_MD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true",
                    help="check links without downloading")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file exists")
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("STEP 1/9  DOWNLOAD DATA")

    if not SOURCES:
        print("SOURCES is empty. Fill in download_data.SOURCES before running.")
        return 2

    print(f"{len(SOURCES)} source(s) registered.\n")
    print("-- link verification --")
    all_ok = True
    for s in SOURCES:
        ok, code, msg = verify_url(s.url)
        s.http_code = code
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {s.name}: {code} {msg}")
        print(f"         {s.url}")
        if not ok and not s.optional:
            all_ok = False
    if not all_ok:
        print("\nAt least one mandatory link is dead. "
              "Replace the source, do NOT put a dead link in the paper.")
        return 1

    if args.verify_only:
        print("\nVerification finished (no download requested).")
        return 0

    print("\n-- download --")
    for s in SOURCES:
        if not download(s, force=args.force) and not s.optional:
            print("Aborting: mandatory download failed.")
            return 1

    md = write_data_sources_md()
    print(f"\nWrote verification report: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
