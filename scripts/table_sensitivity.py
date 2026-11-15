#!/usr/bin/env python3
"""The numbers of the sensitivity analysis (Figure 9 of the paper).

    python scripts/table_sensitivity.py

Reads the runs of the sensitivity experiments (``experiments/*.py``, PUB
setting) that exist under ``results/`` and prints, per configuration, the
median Nash-Sutcliffe efficiency over the evaluation gauges, scored over
2009-2020 as one series: the median over the gauges is taken per seed, then
the mean and standard deviation over the seeds.

  panel  factor                         folder
  a      precipitation product          results/precipitation
  b      temporal resolution of inputs  results/temporal_resolution
  c      additional dynamic inputs      results/dynamic_inputs
  d      river-routing scheme           results/routing_scheme
  e      runoff static descriptors      results/runoff_statics
  g, h   training recipe                results/training_recipe/{two_stage,noroute}

The default configuration (the point every panel is compared with) is the PUB
cell, ``results/settings/pub``.  For the training recipes the scores are also
given by tercile of the evaluation gauges' drainage area (panel h).  Panel f
(resolution of the river network) needs the dataset rebuilt on other meshes
and cannot be reproduced from D3P-JP, which is distributed on the 40 km2 mesh.

Writes ``table_sensitivity.csv`` in ``--out`` (default ``results/``).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import data_root  # noqa: E402
from data import default_splits_path, read_splits_file  # noqa: E402
from metrics import per_gauge_metrics  # noqa: E402
from table_settings import test_series  # noqa: E402

PANELS = [("default", "settings/pub"), ("a precipitation", "precipitation"),
          ("b temporal resolution", "temporal_resolution"), ("c dynamic inputs", "dynamic_inputs"),
          ("d routing scheme", "routing_scheme"), ("e runoff statics", "runoff_statics"),
          ("g training recipe", "training_recipe/two_stage"),
          ("g training recipe", "training_recipe/noroute")]
RECIPES = {"pub": "end-to-end", "two_stage": "two-stage", "noroute": "no routing"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    root = Path(a.data_root) if a.data_root else data_root()
    splits = read_splits_file(default_splits_path(root))
    evaluation = [int(g) for g in splits["evaluation_set"]]
    window = tuple(pd.Timestamp(d) for d in splits["window"])
    area = pd.read_parquet(root / "network" / "nodes.parquet")["drainage_area"].reindex(evaluation)
    lo, hi = area.quantile([1 / 3, 2 / 3])
    tercile = pd.Series(np.where(area <= lo, "T1", np.where(area <= hi, "T2", "T3")), index=area.index)

    rows = []
    for panel, folder in PANELS:
        for p in sorted((res / folder).glob("[0-9][0-9]_*.pkl")):
            m = re.match(r"(\d\d)_(.+)\.pkl$", p.name)
            _, sim, obs = test_series(p)
            days = pd.DatetimeIndex(obs.index).normalize()
            mask = np.asarray((days >= window[0]) & (days <= window[1]))
            nse = per_gauge_metrics(obs, sim, mask=mask)["nse"].reindex(evaluation)
            row = dict(panel=panel, config=m.group(2), seed=int(m.group(1)),
                       n_scored=int(nse.notna().sum()), nse=nse.median())
            if m.group(2) in RECIPES:
                row.update({f"nse_{t}": nse[tercile == t].median() for t in ("T1", "T2", "T3")})
            rows.append(row)
    if not rows:
        raise SystemExit(f"no run found under {res}")
    per_seed = pd.DataFrame(rows)
    cols = [c for c in per_seed.columns if c.startswith("nse")]
    g = per_seed.groupby(["panel", "config"], sort=False)
    table = pd.concat([g[cols].mean(), g[cols].std().add_suffix("_sd"), g["seed"].count().rename("n_seeds"),
                       g["n_scored"].min()], axis=1).reset_index()
    out = Path(a.out or res)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table_sensitivity.csv", index=False, float_format="%.5f")

    print(f"median NSE over the {len(evaluation)} evaluation gauges, {window[0].date()} .. "
          f"{window[1].date()} (mean +- sd over seeds)\n")
    for _, r in table.iterrows():
        sd = f" +- {r.nse_sd:.3f}" if r.n_seeds > 1 else ""
        print(f"  {r.panel:24s}{r.config:26s}{r.nse:.3f}{sd:10s} ({int(r.n_seeds)} seed(s))")
    rec = table[table.config.isin(RECIPES)]
    if len(rec) and "nse_T1" in rec:
        print(f"\npanel h: by tercile of drainage area (below {lo:.0f} km2, {lo:.0f}-{hi:.0f} km2, "
              f"above {hi:.0f} km2)")
        for _, r in rec.iterrows():
            print(f"  {RECIPES[r.config]:12s}" + "".join(f"{r[f'nse_{t}']:8.3f}" for t in ("T1", "T2", "T3")))
    print(f"\nwrote {out}/table_sensitivity.csv")


if __name__ == "__main__":
    main()
