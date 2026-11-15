#!/usr/bin/env python3
"""The default configuration in the holdout settings of the paper (Section 4.2, Table 3).

Every run is ``train.py``'s default configuration (Table 2 of the paper):

  cell   setting                                                          runs per seed
  pub    PUB: 10 folds of whole river basins, trained on 2009-2020; each  1 (10 folds)
         gauge is scored by the model that did not see its river basin
  pubt   PUB-T: the same folds with one four-year block (B1, B2 or B3)    3 (10 folds each)
         withheld from training; each gauge is scored on the withheld
         block by the model that did not see its river basin

    python experiments/settings.py --cells pub --seeds 0-9
    python experiments/settings.py --cells pubt --seeds 0-9
    python experiments/settings.py --dry_run                         # list the commands

Results: ``results/settings/<cell>/<seed>_<cell>[_<block>].pkl`` (series and weights saved).

The two other settings of Table 3 need no training run of their own.  PGB-T
(years withheld, river basin seen) is scored from the PUB-T models, each run
at the gauges of the river basins it was trained on
(``scripts/evaluate_pgbt.py``); PGB (nothing withheld) from the full-data
runs (``experiments/full_data.py``).  ``scripts/table_settings.py`` assembles
the four rows.
"""
import sys
from pathlib import Path

from _common import RESULTS, SAVE, check, dispatch, job, parse_args, pending, seed_range

BLOCKS = ["B1", "B2", "B3"]
CELLS = ["pub", "pubt"]


def cell_configs(cell):
    """``[(exp_name, train.py flags)]`` of one cell; PUB-T has one configuration per block."""
    if cell == "pub":
        return [(cell, SAVE)]
    if cell == "pubt":
        return [(f"{cell}_{b}", ["--holdout_block", b] + SAVE) for b in BLOCKS]
    raise ValueError(f"unknown cell {cell!r}; known: {CELLS}")


def add_args(p):
    p.add_argument("--cells", default=",".join(CELLS),
                   help=f"comma-separated subset of {','.join(CELLS)}")


def main():
    a = parse_args(__doc__, seeds="0-9", add_args=add_args)
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    seeds = list(seed_range(a.seeds))
    jobs = []
    for cell in cells:
        cfgs = cell_configs(cell)
        check(cfgs)
        rdir = Path(a.results_dir) / cell if a.results_dir else RESULTS / "settings" / cell
        cell_jobs = pending(a, [job(a, name, s, rdir, flags) for s in seeds for name, flags in cfgs])
        print(f"[settings] {cell}: {len(cell_jobs)} run(s) pending -> {rdir}", flush=True)
        jobs += cell_jobs
    # seed-major order: seed 0 of every cell first
    jobs.sort(key=lambda oc: int(Path(oc[0]).name[:2]))
    dispatch("settings", jobs, a)


if __name__ == "__main__":
    sys.exit(main())
