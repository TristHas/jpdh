#!/usr/bin/env python3
"""Collect the saved series of several seeds' runs into one store with per-gauge scores.

    python scripts/collect_runs.py results/settings/pub  --out results/collected/pub
    python scripts/collect_runs.py results/settings/pubt --out results/collected/pubt

Reads every ``<seed>_<name>.pkl`` of a results folder (``train.py`` runs made
with ``--save_series``) and writes, in ``--out``:

* ``series.zarr`` -- ``sim(seed, spatial, time)`` and ``obs(spatial, time)``
  in m3/s at every test gauge, and ``fold(spatial)``, the fold whose model
  produced a gauge's series.  For runs with a withheld temporal block (PUB-T,
  ``--holdout_block``), the series of the three block runs of a seed are
  stitched: every day comes from the run that withheld its block, so each value
  is simulated by a model that saw neither the gauge's basin nor the day's period;
* ``scores.csv`` -- per-gauge scores (``src/metrics.py``) per period (the split
  file's window and each block), averaged over seeds;
  ``scores_per_seed.csv`` -- the same before averaging.

These are the series and per-gauge scores behind the figures of the paper;
Table 3 is computed by ``scripts/table_settings.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from data import block_mask, load_time_blocks  # noqa: E402
from metrics import per_gauge_metrics  # noqa: E402


def load_run(path, variant):
    run = pd.read_pickle(path)
    cfg = run["config"]
    ystd = float(cfg["y_std"])
    sim, obs, fold = [], [], {}
    for fr in run["fold_results"]:
        ev = fr[variant]
        if "ote" not in ev:
            raise SystemExit(f"{path}: no saved series (train with --save_series)")
        sim.append(ev["ote"] * ystd)
        obs.append(ev["yte"] * ystd)
        fold.update({int(g): int(fr["fold"]) for g in ev["yte"].columns})
    sim, obs = pd.concat(sim, axis=1), pd.concat(obs, axis=1)
    sim.columns = obs.columns = [int(c) for c in obs.columns]
    return cfg, sim, obs, pd.Series(fold)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="swa")
    a = ap.parse_args()

    paths = sorted(Path(a.results_dir).glob("[0-9][0-9]_*.pkl"))
    if not paths:
        raise SystemExit(f"no <seed>_<name>.pkl in {a.results_dir}")
    runs = {}
    for p in paths:
        cfg, sim, obs, fold = load_run(p, a.variant)
        block = cfg.get("holdout_block", "none")
        runs.setdefault(int(cfg["seed"]), {})[block] = (cfg, sim, obs, fold)
        print(f"{p.name}: seed {cfg['seed']}, block {block}, {sim.shape[1]} gauges", flush=True)
    cfg0 = next(iter(next(iter(runs.values())).values()))[0]
    splits = json.loads(Path(cfg0["splits_file"]).read_text())
    blocks_file = cfg0.get("time_blocks_file") or Path(cfg0["splits_file"]).with_name("time_blocks.json")
    blocks = load_time_blocks(blocks_file)

    seeds = sorted(runs)
    sims, obs_ref, fold_ref, gauges, times = [], None, None, None, None
    for s in seeds:
        by_block = runs[s]
        if "none" in by_block:
            if len(by_block) > 1:
                raise SystemExit(f"seed {s}: mixes runs with and without --holdout_block")
            _, sim, obs, fold = by_block["none"]
        else:
            missing = sorted(set(blocks) - set(by_block))
            if missing:
                raise SystemExit(f"seed {s}: block run(s) {missing} missing")
            _, sim0, obs, fold = next(iter(by_block.values()))
            sim = pd.DataFrame(np.nan, index=sim0.index, columns=sim0.columns)
            for b, (_, sb, _, fb) in by_block.items():
                m = block_mask(sb.index, blocks[b]).numpy()
                sim.loc[m] = sb.loc[m, sim.columns].values
                assert fb.reindex(fold.index).equals(fold), "fold assignment differs between block runs"
        if gauges is None:
            gauges = [g for g in splits.get("evaluation_set", sim.columns) if g in sim.columns]
            times, obs_ref, fold_ref = sim.index, obs[gauges], fold.reindex(gauges)
        else:
            assert sim.index.equals(times), f"seed {s}: different time axis"
            d = np.nanmax(np.abs(obs[gauges].values - obs_ref.values))
            assert d < 1e-3, f"seed {s}: observations differ ({d})"
        sims.append(sim[gauges].values.T.astype(np.float32))

    window = pd.DatetimeIndex(times)
    w0, w1 = (pd.Timestamp(d) for d in splits["window"])
    in_window = np.asarray((window.normalize() >= w0) & (window.normalize() <= w1))
    per_seed = []
    for s, arr in zip(seeds, sims):
        sim = pd.DataFrame(arr.T, index=window, columns=gauges)
        for period, mask in [("window", in_window)] + [
                (b, in_window & block_mask(window, bd).numpy()) for b, bd in blocks.items()]:
            tab = per_gauge_metrics(obs_ref, sim, mask=mask)
            tab.insert(0, "period", period)
            tab.insert(0, "seed", s)
            per_seed.append(tab)
    per_seed = pd.concat(per_seed)
    per_seed.index.name = "spatial"
    mean = per_seed.drop(columns="seed").groupby(["period", per_seed.index]).mean()
    mean["lag"] = per_seed.groupby(["period", per_seed.index])["lag"].median()
    mean["n_seeds"] = per_seed.groupby(["period", per_seed.index])["nse"].count()
    mean = mean.reset_index().set_index("spatial")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({"sim": (("seed", "spatial", "time"), np.stack(sims)),
                     "obs": (("spatial", "time"), obs_ref.values.T.astype(np.float32)),
                     "fold": (("spatial",), fold_ref.values.astype(np.int16))},
                    coords={"seed": np.array(seeds, np.int16), "spatial": np.array(gauges, np.int64),
                            "time": window.values})
    setting = "PUB-T (stitched block runs)" if "none" not in runs[seeds[0]] else "PUB"
    ds.attrs = {"title": "Evaluation series on the D3P-JP river network", "setting": setting,
                "configuration": json.dumps({k: cfg0.get(k) for k in (
                    "irf_fn", "routing_features", "routing_bounds", "routing_init", "dynamic_keys",
                    "static_runoff_keys", "routing_lr", "runoff_lr", "n_epoch", "variants")}),
                "time_convention": "12Z days: the label YYYY-MM-DD 12:00 UTC covers [12:00 UTC, next day 12:00 UTC)"}
    ds["sim"].attrs.update(units="m3 s-1", long_name="simulated discharge")
    ds["obs"].attrs.update(units="m3 s-1", long_name="observed discharge")
    enc = {"sim": {"chunks": (1, 128, len(window))}, "obs": {"chunks": (128, len(window))}}
    ds.to_zarr(out / "series.zarr", mode="w", encoding=enc, consolidated=True)
    mean.to_csv(out / "scores.csv", float_format="%.5f")
    per_seed.to_csv(out / "scores_per_seed.csv", float_format="%.5f")
    print(f"\n{setting}, {len(seeds)} seed(s), {len(gauges)} gauges -> {out}")
    for period in ["window"] + list(blocks):
        med = per_seed[per_seed.period == period].groupby("seed")[["nse", "kge"]].median()
        print(f"  {period:6s} median NSE {med.nse.mean():.3f} +- {med.nse.std(ddof=1) if len(med) > 1 else 0:.3f}, "
              f"median KGE' {med.kge.mean():.3f} (mean +- sd over seeds of the per-seed medians)")


if __name__ == "__main__":
    main()
