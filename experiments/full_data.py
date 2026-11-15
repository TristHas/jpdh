#!/usr/bin/env python3
"""Full-data runs: the model of the released simulation (D3P-JP-Q) and the PGB setting.

The default configuration trained on *all* the training gauges over
2009-2020 -- no river basin and no year withheld.  The split file has a single
fold: ``train`` = ``training_pool``, ``val`` = ``training_pool`` (monitoring
only), ``test`` = ``evaluation_set``; so the run's scores are the fit to the
training records at the training gauges and are *not* a generalization score:
they are the PGB row of Table 3 (``scripts/table_settings.py``).  The
normalisation set is the dataset's, so ``y_std`` is the PUB runs'.  The file
is derived deterministically from the dataset's split file and written to
``results/full_data/pub_folds_fulldata.json``.

Ten seeds, saved series and weights.  Seed 0 is the model of D3P-JP-Q:
``scripts/export_weights.py`` turns a run into a model file and
``scripts/simulate.py`` runs it over every catchment.

    python experiments/full_data.py --seeds 0-9
    python experiments/full_data.py --seeds 0-0        # the released simulation's model only
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _common import RESULTS, launch
from data import default_splits_path

FULLDATA_SPLITS = RESULTS / "full_data" / "pub_folds_fulldata.json"


def derive_fulldata_splits(src: Path, dst: Path = FULLDATA_SPLITS) -> Path:
    """Write the single-fold full-data split file from the dataset's file (idempotent)."""
    base = json.loads(src.read_text())
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if dst.exists() and json.loads(dst.read_text()).get("source_sha256") == sha:
        return dst
    pool = [int(g) for g in base["training_pool"]]
    evaluation = [int(g) for g in base["evaluation_set"]]
    assert set(pool) <= set(evaluation)
    out = {k: v for k, v in base.items() if k != "basin_folds"}
    out["n_folds"] = 1
    out["folds"] = [{"fold": 0, "train": pool, "val": pool, "test": evaluation}]
    out["protocol"] = ("Full-data run: one fold trained on the whole training pool over the whole "
                       "window; val = training pool (monitoring only), test = evaluation set "
                       "(in-sample at training-pool gauges, out-of-training at the others). "
                       "Not an evaluation protocol; normalisation set unchanged.")
    out["source_sha256"] = sha
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    print(f"[full_data] wrote {dst}: train {len(pool)}, test {len(evaluation)}", flush=True)
    return dst


if __name__ == "__main__":
    splits = derive_fulldata_splits(default_splits_path())
    # --inference_batch_size 1: the val set is the whole training pool, whose
    # subgraph is ~10x a fold's (evaluation batching only -- the series are identical)
    CONFIGS = [("full_data", ["--splits_file", str(splits), "--save_series", "--save_weights",
                              "--inference_batch_size", "1"])]
    launch("full_data", CONFIGS, __doc__, seeds="0-9")
