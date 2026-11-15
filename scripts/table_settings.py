#!/usr/bin/env python3
"""Table 3 of the paper: accuracy in the four evaluation settings.

    python scripts/table_settings.py

  setting  what is withheld from training      runs it is read from
  PGB      nothing (the released simulation)   results/full_data/<seed>_full_data.pkl
  PGB-T    the scored four-year block          results/settings/pgbt/<seed>_pgbt_<block>.csv
  PUB      the gauge's river basin             results/settings/pub/<seed>_pub.pkl
  PUB-T    both                                results/settings/pubt/<seed>_pubt_<block>.pkl

(``experiments/full_data.py``, ``scripts/evaluate_pgbt.py`` and
``experiments/settings.py``.)  A setting whose runs are missing is left out.

Estimator, the same for the four rows: every gauge is scored on each
four-year block (B1 2009-2012, B2 2013-2016, B3 2017-2020; gauges with at
least 364 valid days in the block); the median over the gauges is taken per
seed and block, averaged over the three blocks, and reported as mean +- standard
deviation over the seeds.  In PGB-T a gauge's score on a block is the mean of
the scores of the nine fold models that were trained on its river basin.
Populations: all evaluation gauges, and the unregulated ones (the training
gauges, with no dam upstream).

Writes ``table_settings.csv`` (the table) and ``table_settings_per_seed.csv``
(setting, population, seed, block) in ``--out`` (default ``results/``).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import data_root  # noqa: E402
from data import (block_mask, default_splits_path, load_time_blocks, read_splits_file,  # noqa: E402
                  time_blocks_path)
from metrics import per_gauge_metrics  # noqa: E402

SETTINGS = ["PGB", "PGB-T", "PUB", "PUB-T"]
METRICS = ["nse", "kge"]


def test_series(path, variant="swa"):
    """``(config, sim, obs)`` of a run: the test series of its folds side by side, in m3/s."""
    run = pd.read_pickle(path)
    cfg = run["config"]
    ystd = float(cfg["y_std"])
    sim, obs = [], []
    for fr in run["fold_results"]:
        ev = fr[variant]
        if "ote" not in ev:
            raise SystemExit(f"{path}: no saved series (train with --save_series)")
        sim.append(ev["ote"] * ystd)
        obs.append(ev["yte"] * ystd)
    sim, obs = pd.concat(sim, axis=1), pd.concat(obs, axis=1)
    sim.columns = obs.columns = [int(c) for c in obs.columns]
    return cfg, sim, obs


def block_scores(path, blocks, window, only=None):
    """Per-gauge scores of a run on each block (``only``: a single block): rows (block, gauge)."""
    cfg, sim, obs = test_series(path)
    days = pd.DatetimeIndex(obs.index).normalize()
    in_window = np.asarray((days >= window[0]) & (days <= window[1]))
    out = []
    for b, bounds in blocks.items():
        if only is not None and b != only:
            continue
        tab = per_gauge_metrics(obs, sim, mask=in_window & block_mask(obs.index, bounds).numpy())[METRICS]
        tab.insert(0, "block", b)
        out.append(tab)
    out = pd.concat(out)
    out.insert(0, "seed", int(cfg["seed"]))
    out.index.name = "spatial"
    return cfg, out.reset_index()


def collect(a, blocks, window):
    """Long table (setting, seed, block, spatial, nse, kge), one row per scored gauge."""
    parts = []

    def runs(folder, pattern):
        found = []
        for p in sorted(Path(folder).glob("[0-9][0-9]_*")):
            m = re.match(pattern, p.name)
            if m:
                found.append((p, m))
        return found

    for p, _ in runs(a.full, r"\d\d_full_data\.pkl$"):
        cfg, t = block_scores(p, blocks, window)
        parts.append(t.assign(setting="PGB"))
    for p, _ in runs(a.pub, r"\d\d_pub\.pkl$"):
        cfg, t = block_scores(p, blocks, window)
        if cfg.get("holdout_block", "none") != "none":
            raise SystemExit(f"{p}: trained with a withheld block, not a PUB run")
        parts.append(t.assign(setting="PUB"))
    for p, m in runs(a.pubt, r"\d\d_pubt_(B\d)\.pkl$"):
        cfg, t = block_scores(p, blocks, window, only=m.group(1))
        if cfg.get("holdout_block") != m.group(1):
            raise SystemExit(f"{p}: trained with --holdout_block {cfg.get('holdout_block')}")
        parts.append(t.assign(setting="PUB-T"))
    for p, _ in runs(a.pgbt, r"\d\d_pgbt_B\d\.csv$"):
        t = pd.read_csv(p)
        g = t.groupby(["seed", "block", "spatial"])
        mean = g[METRICS].mean().reset_index()
        n_models = g["fold"].nunique().to_numpy()
        if n_models.min() != n_models.max():       # a run with fewer than all its folds
            print(f"note: {p.name}: gauges scored by {n_models.min()} to {n_models.max()} fold models")
        parts.append(mean.assign(setting="PGB-T"))
    if not parts:
        raise SystemExit("no run found; see the module docstring for the expected folders")
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    res = REPO / "results"
    ap.add_argument("--full", default=str(res / "full_data"))
    ap.add_argument("--pub", default=str(res / "settings" / "pub"))
    ap.add_argument("--pubt", default=str(res / "settings" / "pubt"))
    ap.add_argument("--pgbt", default=str(res / "settings" / "pgbt"))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default=str(res))
    a = ap.parse_args()

    root = Path(a.data_root) if a.data_root else data_root()
    splits_path = default_splits_path(root)
    splits = read_splits_file(splits_path)
    blocks = load_time_blocks(time_blocks_path(splits_path, splits))
    window = tuple(pd.Timestamp(d) for d in splits["window"])
    pops = {"all gauges": [int(g) for g in splits["evaluation_set"]],
            "unregulated gauges": [int(g) for g in splits["training_pool"]]}

    long = collect(a, blocks, window)
    long = long[long.spatial.isin(pops["all gauges"])]
    rows = []
    for (setting, seed, block), t in long.groupby(["setting", "seed", "block"]):
        for pop, ids in pops.items():
            sub = t[t.spatial.isin(ids)]
            rows.append(dict(setting=setting, population=pop, seed=seed, block=block,
                             n_scored=int(sub.nse.notna().sum()),
                             nse=sub.nse.median(), kge=sub.kge.median()))
    per_seed = pd.DataFrame(rows)
    n_blocks = per_seed.groupby(["setting", "population", "seed"])["block"].nunique()
    incomplete = n_blocks[n_blocks != len(blocks)]
    if len(incomplete):
        print(f"note: seeds without the {len(blocks)} blocks are left out: "
              f"{sorted({(s, int(k)) for s, _, k in incomplete.index})}")
        keep = n_blocks[n_blocks == len(blocks)].index
        per_seed = per_seed.set_index(["setting", "population", "seed"]).loc[keep].reset_index()
    seed_mean = per_seed.groupby(["setting", "population", "seed"])[METRICS].mean()
    table = seed_mean.groupby(["setting", "population"]).agg(["mean", "std", "count"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table = table.rename(columns={"nse_count": "n_seeds"}).drop(columns="kge_count")
    table["n_gauges"] = [len(pops[p]) for _, p in table.index]
    table = table.reindex([s for s in SETTINGS if s in table.index.get_level_values(0)], level=0)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "table_settings_per_seed.csv", index=False, float_format="%.5f")
    table.to_csv(out / "table_settings.csv", float_format="%.5f")

    def cell(setting, pop, m):
        r = table.loc[(setting, pop)]
        sd = f" +- {r[f'{m}_std']:.3f}" if r["n_seeds"] > 1 else ""
        return f"{r[f'{m}_mean']:.3f}{sd}"

    names = list(pops)
    head = "".join(f"{f'{label}, {p} (n={len(pops[p])})':>42s}" for label in ("NSE", "KGE'") for p in names)
    print(f"\n{'Setting':8s}{'seeds':>6s}{head}")
    for setting in table.index.get_level_values(0).unique():
        cells = "".join(f"{cell(setting, p, m):>42s}" for m in METRICS for p in names)
        print(f"{setting:8s}{int(table.loc[(setting, names[0]), 'n_seeds']):>6d}{cells}")
    print(f"\nmedian over the gauges per seed and block, mean over the {len(blocks)} blocks, "
          f"mean +- sd over the seeds\nwrote {out}/table_settings.csv and table_settings_per_seed.csv")


if __name__ == "__main__":
    main()
