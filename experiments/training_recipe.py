#!/usr/bin/env python3
"""Training recipe: end-to-end against two-stage training and no routing (Figure 9g-h).

Same data, folds (PUB), architecture, loss and schedule as the default
configuration; the end-to-end arm is the ``pub`` cell of
``experiments/settings.py`` and is not repeated here.

  arm        what                                                          output
  lumped     stage 1 of the two-stage recipe (experiments/train_lumped.py): results/training_recipe/lumped/
             the runoff module trained as a lumped model against each      <seed>_lumped_fold<k>.pt
             training gauge, one checkpoint per (seed, fold)
  two_stage  stage 2: the stage-1 runoff module frozen, the routing-       results/training_recipe/two_stage/
             parameter module calibrated alone through the routing
  noroute    no routing: the runoff of all upstream catchments reaches     results/training_recipe/noroute/
             the gauge on the same day (train.py --irf_fn instantaneous)

    python experiments/training_recipe.py --arms lumped --seeds 0-9
    python experiments/training_recipe.py --arms two_stage --seeds 0-9      # needs the lumped checkpoints
    python experiments/training_recipe.py --arms noroute --seeds 0-9
"""
import sys
from pathlib import Path

from _common import REPO, RESULTS, SAVE, check, dispatch, job, parse_args, pending, seed_range

ROOT = RESULTS / "training_recipe"
LUMPED_DIR = ROOT / "lumped"
LUMPED = LUMPED_DIR / "{seed:02d}_lumped_fold{fold}.pt"
ARMS = ["lumped", "two_stage", "noroute"]
CONFIGS = {
    "two_stage": ("two_stage", ["--runoff_init", str(LUMPED), "--freeze_runoff"] + SAVE),
    "noroute": ("noroute", ["--irf_fn", "instantaneous"] + SAVE),
}


def add_args(p):
    p.add_argument("--arms", default=",".join(ARMS),
                   help="comma-separated subset of " + ",".join(ARMS))
    p.add_argument("--n_folds", type=int, default=10,
                   help="folds per seed (the last fold's checkpoint marks a lumped seed as done)")


def missing_stage1(seeds, n_folds):
    paths = [Path(str(LUMPED).format(seed=s, fold=f)) for s in seeds for f in range(n_folds)]
    return [p for p in paths if not p.exists()]


def main():
    a = parse_args(__doc__, seeds="0-9", add_args=add_args)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    unknown = [s for s in arms if s not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {ARMS}")
    check(list(CONFIGS.values()))
    seeds = list(seed_range(a.seeds))
    for arm in [s for s in ARMS if s in arms]:
        if arm == "lumped":
            # train_lumped.py skips existing fold files, so the last fold's
            # file is the seed's done marker
            jobs = []
            for seed in seeds:
                out = LUMPED_DIR / f"{seed:02d}_lumped_fold{a.n_folds - 1}.pt"
                cmd = [sys.executable, "-u", str(REPO / "experiments" / "train_lumped.py"),
                       "--seed", str(seed), "--results_dir", str(LUMPED_DIR),
                       "--device", a.device]
                jobs.append((out, cmd + list(a.extra)))
        else:
            name, flags = CONFIGS[arm]
            jobs = [job(a, name, seed, ROOT / arm, flags) for seed in seeds]
        jobs = pending(a, jobs)
        if arm == "two_stage" and jobs and not a.dry_run:
            missing = missing_stage1(seeds, a.n_folds)
            if missing:
                raise SystemExit(f"two_stage: {len(missing)} stage-1 checkpoint(s) missing, e.g. "
                                 f"{missing[0]}; run --arms lumped first")
        print(f"[training_recipe] {arm}: {len(jobs)} run(s) pending", flush=True)
        dispatch(f"training_recipe_{arm}", jobs, a)


if __name__ == "__main__":
    main()
