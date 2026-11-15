#!/usr/bin/env python3
"""PGB-T scores: the PUB-T models run at the gauges of the river basins they were trained on.

    python scripts/evaluate_pgbt.py results/settings/pubt --out results/settings/pgbt

In the PGB-T setting (prediction in gauged basins with a temporal holdout) a
gauge is scored on a four-year block by models that saw its river basin but
not that block.  These are the PUB-T models: the run ``<seed>_pubt_<block>.pkl``
holds ten fold models trained without ``<block>``, and nine of them were
trained on the river basin of any given gauge.  For every run of the folder
(``experiments/settings.py --cells pubt``, which saves the weights), each fold
model simulates the evaluation gauges *outside* its held-out fold and is
scored on the withheld block.

Output: one ``<seed>_pgbt_<block>.csv`` per run in ``--out`` (runs whose file
exists are skipped), with one row per (fold model, gauge) and the scores of
``src/metrics.py`` on the withheld block.  ``scripts/table_settings.py``
averages the nine models of each gauge.  One run takes a few minutes on a GPU;
``--shard i/n`` splits the runs across processes.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from config import data_root  # noqa: E402
from data import (block_mask, data_loading, default_splits_path, load_time_blocks,  # noqa: E402
                  read_splits_file, time_blocks_path, time_index)
from inference import simulate  # noqa: E402
from metrics import per_gauge_metrics  # noqa: E402
from weights import check_norm, model_from_payload, run_payloads  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help="folder of the PUB-T runs (<seed>_pubt_<block>.pkl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_groups", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="windows per forward pass (training evaluates test gauges with 1)")
    ap.add_argument("--shard", default=None, help="'i/n': every n-th run, offset i")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in sorted(Path(a.runs).glob("[0-9][0-9]_*.pkl")):
        m = re.match(r"(\d\d)_(.+)_(B\d)\.pkl$", p.name)
        if m:
            runs.append((p, int(m.group(1)), m.group(3), out / f"{m.group(1)}_pgbt_{m.group(3)}.csv"))
    if not runs:
        raise SystemExit(f"no <seed>_<name>_<block>.pkl in {a.runs}")
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        runs = runs[i::n]
    if not a.force:
        runs = [r for r in runs if not r[3].exists()]
    print(f"{len(runs)} run(s) to evaluate -> {out}", flush=True)
    if not runs:
        return

    root = Path(a.data_root) if a.data_root else data_root()
    splits_path = default_splits_path(root)
    splits = read_splits_file(splits_path)
    evaluation = [int(g) for g in splits["evaluation_set"]]
    blocks = load_time_blocks(time_blocks_path(splits_path, splits))
    w0, w1 = (pd.Timestamp(d) for d in splits["window"])
    data, spec, obs = None, None, None

    for path, seed, block, dst in runs:
        t0 = time.time()
        payloads = run_payloads(path)
        held = {p["meta"]["holdout_block"] for p in payloads}
        if held != {block}:
            raise SystemExit(f"{path}: trained with --holdout_block {sorted(held)}, not {block}")
        if data is None:
            spec = payloads[0]["spec"]
            data = data_loading(dynamic_keys=spec["dynamic_keys"],
                                static_runoff_keys=spec["static_runoff_keys"],
                                routing_static_var=spec["routing_features"], root=root)
            y, y_std = data[2], float(data[5].values)
            obs = pd.DataFrame(y.values[0].detach().cpu().numpy().T * y_std,
                               index=pd.DatetimeIndex(time_index(y)),
                               columns=[int(s) for s in y.coords["spatial"]])
        rows = []
        for payload in payloads:
            if payload["spec"] != spec:
                raise SystemExit(f"{path}: a different configuration from the first run")
            check_norm(payload, data[3], data[4], data[5])
            fold = payload["meta"]["fold"]
            test = set(payload["meta"]["test_gauges"])
            targets = [g for g in evaluation if g not in test]
            model = model_from_payload(payload, device=a.device)
            times, Q, _ = simulate(model, spec, data, targets, a.device, n_groups=a.n_groups,
                                   batch_size=a.batch_size, log=lambda s: None,
                                   y_std=payload["norm"]["y_std"])
            del model
            sim = pd.DataFrame(Q, index=times, columns=targets)
            days = pd.DatetimeIndex(times).normalize()
            mask = np.asarray((days >= w0) & (days <= w1)) & block_mask(times, blocks[block]).numpy()
            tab = per_gauge_metrics(obs.loc[times, targets], sim, mask=mask)
            tab.insert(0, "fold", fold)
            rows.append(tab)
        scores = pd.concat(rows)
        scores.index.name = "spatial"
        scores.insert(0, "block", block)
        scores.insert(0, "seed", seed)
        scores.to_csv(dst, float_format="%.5f")
        per_gauge = scores.groupby(level=0)["nse"].mean()
        print(f"{path.name}: {len(payloads)} fold models, {per_gauge.notna().sum()} gauges scored on "
              f"{block}, median NSE {per_gauge.median():.3f} ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
