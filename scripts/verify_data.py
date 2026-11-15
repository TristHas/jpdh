#!/usr/bin/env python3
"""Check a downloaded dataset: files, checksums, coordinates and the PUB folds.

    python scripts/verify_data.py                   # $JPDH_DATA_ROOT or ./data
    python scripts/verify_data.py --no-checksums    # skip the per-file SHA-256 pass

Every archive carries ``checksums/<archive>.json`` (SHA-256 of each file it
extracted); they are all verified.  Then the stores are opened and their
coordinates checked against the network, and the default configuration's
inputs are checked to be complete.  Exits non-zero on the first failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from config import DEFAULT_DYNAMIC_KEYS, DEFAULT_STATIC_RUNOFF_KEYS, data_root, \
    expand_dynamic_keys, expand_static_keys  # noqa: E402

failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)
    return cond


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--no-checksums", action="store_true")
    a = ap.parse_args()
    root = Path(a.data_root) if a.data_root else data_root()
    print(f"dataset: {root}")
    if not check((root / "network" / "nodes.parquet").exists(), "core archive present"):
        print("run: python scripts/download_data.py")
        return 1

    if not a.no_checksums:
        print("\nchecksums")
        for cfile in sorted((root / "checksums").glob("*.json")):
            sums = json.loads(cfile.read_text())
            bad = [p for p, h in sums.items() if not (root / p).exists() or sha256(root / p) != h]
            check(not bad, f"{cfile.stem}: {len(sums)} files" + (f", {len(bad)} differ, e.g. {bad[0]}"
                                                                  if bad else ""))

    print("\nnetwork and statics")
    nodes = pd.read_parquet(root / "network" / "nodes.parquet")
    edges = pd.read_parquet(root / "network" / "edges.parquet")
    check(nodes.index.is_unique, f"{len(nodes)} nodes, unique ids")
    check(edges["source"].isin(nodes.index).all() and edges["target"].isin(nodes.index).all(),
          f"{len(edges)} edges between known nodes")
    run = pd.read_parquet(root / "statics" / "runoff_statics.parquet")
    rout = pd.read_parquet(root / "statics" / "routing_statics.parquet")
    check(run.index.equals(nodes.index) and rout.index.equals(nodes.index),
          "statics indexed like the nodes")
    cols = expand_static_keys(DEFAULT_STATIC_RUNOFF_KEYS)
    if check(set(cols) <= set(run.columns), f"default runoff statics present ({len(cols)} columns)"):
        nan = float(run[cols].isna().mean().mean())
        check(nan < 0.05, f"default runoff statics {100 * nan:.2f} % missing (set to the mean)")

    print("\nforcing and observations")
    f = xr.open_zarr(root / "daily" / "forcing.zarr")
    check(np.array_equal(f.spatial.values, nodes.index.values), "forcing on the network's nodes")
    t = pd.DatetimeIndex(f.time.values)
    check(bool((t.hour == 12).all()) and t.is_monotonic_increasing and
          bool((np.diff(t.values) == np.timedelta64(1, "D")).all()),
          f"daily 12Z axis {t[0].date()} .. {t[-1].date()} ({len(t)} days)")
    need = expand_dynamic_keys(DEFAULT_DYNAMIC_KEYS)
    if check(set(need) <= set(f.data_vars), f"default forcing variables present ({len(need)})"):
        nan = float(f[need].to_array().isnull().mean())
        check(nan < 1e-3, f"default forcing {100 * nan:.4f} % missing")
    panels = root / "daily" / "forcing_panels.zarr"
    if panels.exists():
        fp = xr.open_zarr(panels)
        check(np.array_equal(fp.spatial.values, f.spatial.values) and
              np.array_equal(fp.time.values, f.time.values),
              f"panels store aligned with the forcing ({len(fp.data_vars)} variables)")
    else:
        print("  --    panels archive not installed (needed only by the sensitivity experiments)")
    q = xr.open_zarr(root / "daily" / "discharge.zarr")
    check(np.isin(q.spatial.values, nodes.index.values).all(),
          f"discharge at {q.sizes['spatial']} gauges, all on network nodes")
    check(np.array_equal(q.time.values, f.time.values), "discharge on the forcing's day axis")

    print("\nfolds")
    s = json.loads((root / "splits" / "pub_folds_10.json").read_text())
    ev, pool = set(s["evaluation_set"]), set(s["training_pool"])
    tests = [set(fd["test"]) for fd in s["folds"]]
    check(len(s["folds"]) == 10, "10 folds")
    check(set().union(*tests) == ev and sum(map(len, tests)) == len(ev),
          f"every one of the {len(ev)} evaluation gauges is tested exactly once")
    check(all(set(fd["train"]) <= pool and not set(fd["train"]) & set(fd["test"]) for fd in s["folds"]),
          f"training sets inside the {len(pool)}-gauge training pool, disjoint from their test set")
    check(set(s["gauges_with_discharge"]) <= set(int(v) for v in q.spatial.values),
          "normalisation gauges have discharge records")
    check(ev == set(int(v) for v in q.spatial.values),
          "the discharge store holds the evaluation gauges and no other")

    print(f"\n{'all checks passed' if not failures else f'{len(failures)} check(s) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
