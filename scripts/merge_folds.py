#!/usr/bin/env python3
"""Merge per-fold result files from a parallel run into a single run file.

``train.py --folds`` seeds each fold from its original index, so N parallel
single-fold runs produce exactly what one sequential run would have.  This
merges them back into the same structure a sequential run writes.

    python train.py --exp_name default --folds 0,1,2,3,4 --device cuda:0 &
    python train.py --exp_name default --folds 5,6,7,8,9 --device cuda:1
    python scripts/merge_folds.py results/00_default_folds*.pkl -o results/00_default.pkl
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ALL_VARIANTS = ("final", "best", "swa")

#: config keys that legitimately differ between shards (devices, times)
IGNORE = {"folds", "device", "exp_name", "results_dir",
          "started_at", "finished_at", "wall_time_s", "fold_wall_time_s"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="per-fold pickles (globs allowed)")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    paths = sorted({q for pat in args.inputs for q in glob.glob(pat)})
    if not paths:
        raise SystemExit("no input files matched")

    shards = [pd.read_pickle(q) for q in paths]

    # every shard must share the same experimental configuration
    base = {k: v for k, v in shards[0]["config"].items() if k not in IGNORE}
    for q, s in zip(paths[1:], shards[1:]):
        cur = {k: v for k, v in s["config"].items() if k not in IGNORE}
        diff = {k for k in base | cur if str(base.get(k)) != str(cur.get(k))}
        if diff:
            raise SystemExit(f"config mismatch in {q}: {sorted(diff)}")

    order = []
    for q, s in zip(paths, shards):
        idx = s["config"].get("folds")
        idx = ([int(i) for i in str(idx).split(",")] if idx is not None
               else list(range(len(s["fold_results"]))))
        if len(idx) != len(s["fold_results"]):
            raise SystemExit(f"{q}: {len(idx)} indices vs "
                             f"{len(s['fold_results'])} fold results")
        order.extend(zip(idx, s["fold_results"]))

    seen = [i for i, _ in order]
    if len(set(seen)) != len(seen):
        dup = sorted({i for i in seen if seen.count(i) > 1})
        raise SystemExit(f"fold(s) present in more than one shard: {dup}")
    if sorted(seen) != list(range(len(seen))):
        print(f"  warning: folds {sorted(seen)} are not a complete 0..N range")

    fold_results = [fr for _, fr in sorted(order, key=lambda t: t[0])]

    # a run may report a subset of the variants (train.py --variants)
    variants = [v for v in ALL_VARIANTS if v in fold_results[0]]

    summary = {}
    for v in variants:
        nse_val = pd.concat([fr[v]["nse_val"] for fr in fold_results])
        nse_te = pd.concat([fr[v]["nse_te"] for fr in fold_results])
        summary[v] = dict(val_median=float(np.nanmedian(nse_val.values)),
                          val_mean=float(np.nanmean(nse_val.values)),
                          te_median=float(np.nanmedian(nse_te.values)),
                          te_mean=float(np.nanmean(nse_te.values)))

    merged = dict(shards[0])
    merged["config"] = {k: v for k, v in shards[0]["config"].items()
                        if k != "folds"}
    merged["fold_results"] = fold_results
    merged["summary"] = summary

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(merged, out)

    print(f"merged {len(paths)} shards -> {len(fold_results)} folds -> {out}")
    for v in variants:
        print(f"  {v:6s} val med={summary[v]['val_median']:.4f}  "
              f"te med={summary[v]['te_median']:.4f}")


if __name__ == "__main__":
    main()
