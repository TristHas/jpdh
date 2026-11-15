"""Shared plumbing for the experiment scripts.

Each script next to this one is one experiment of the paper: a table of the
configurations it varies off the default configuration (``train.py``'s
defaults), expressed as ``train.py`` flags, and a call to ``launch``.  Nothing
here computes -- the model, the training and the outputs all live in
``train.py`` and ``src/``.

Ways to run a script's pending jobs (anything whose output pickle already
exists is skipped, so re-running resumes):

* directly, one after the other on ``--device``;
* ``--shard i/n``: take every n-th job, offset i, so n processes on n GPUs
  cover the list between them;
* ``--dry_run``: print the commands (e.g. to submit them to a scheduler).

Any flag a script does not recognise is passed through to ``train.py``.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config import (DEFAULT_IRF, KERNEL_BOUNDS, expand_dynamic_keys,  # noqa: E402
                    expand_static_keys)
from model import IRF_CHOICES                                         # noqa: E402

#: results root of every experiment (``results/<experiment>/<seed>_<config>.pkl``)
RESULTS = Path(os.environ.get("JPDH_RESULTS_ROOT", REPO / "results"))

#: the default forcing pair, for the panels that vary or extend it
BASE_PRCP, BASE_TEMP = "gauge_prcp_4h_bin", "msm_a_temp_4h_bin"
BASE_DYNAMIC = f"{BASE_PRCP},{BASE_TEMP}"

#: what every reported run stores on top of its scores
SAVE = ["--save_series", "--save_weights"]


def dyn(*keys):
    return ["--dynamic_vars", ",".join(keys)]


def flags_dict(args):
    """``{flag: value}`` of a train.py flag list; a flag without a value (``--save_series``)
    maps to True, so a valueless flag never shifts the pairing of the ones after it."""
    out, i = {}, 0
    while i < len(args):
        tok = str(args[i])
        nxt = str(args[i + 1]) if i + 1 < len(args) else None
        if tok.startswith("--") and nxt is not None and not nxt.startswith("--"):
            out[tok], i = nxt, i + 2
        else:
            out[tok], i = True, i + 1
    return out


def check(configs):
    """Fail on a mistyped variable or kernel now rather than after the data load."""
    for name, args in configs:
        a = flags_dict(args)
        if "--dynamic_vars" in a:
            expand_dynamic_keys(a["--dynamic_vars"].split(","))
        if "--static_runoff_vars" in a:
            expand_static_keys(a["--static_runoff_vars"].split(","))
        irf = a.get("--irf_fn", DEFAULT_IRF)
        if irf not in IRF_CHOICES:
            raise ValueError(f"{name}: unknown --irf_fn {irf!r}; valid {IRF_CHOICES}")


def kernel_flags(kernel, features):
    """``train.py`` flags of a routing kernel with its default bounds and the given descriptors."""
    flags = ["--irf_fn", kernel, "--routing_static_vars", ",".join(features)]
    if kernel in KERNEL_BOUNDS:
        flags += ["--routing_bounds", KERNEL_BOUNDS[kernel]]
    return flags


def parse_args(description, seeds="0-9", add_args=None):
    """The shared flags; ``add_args(parser)`` lets a script add its own."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    if add_args is not None:
        add_args(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default=seeds,
                   help="inclusive range, e.g. 0-9; the paper reports the seeds given as default")
    p.add_argument("--results_dir", default=None,
                   help="default: results/<experiment name>")
    p.add_argument("--shard", default=None,
                   help="'i/n': take every n-th job, offset i, so n processes "
                        "on n GPUs cover the list between them")
    p.add_argument("--force", action="store_true",
                   help="re-run configurations whose output already exists")
    p.add_argument("--dry_run", action="store_true",
                   help="print the commands instead of running them")
    # anything this parser does not recognise is passed through to train.py,
    # so e.g. `--max_folds 1 --n_epoch 2` works without being redeclared here
    args, extra = p.parse_known_args()
    if extra and extra[0] == "--":          # `script.py ... -- --flag` form
        extra = extra[1:]
    args.extra = extra
    return args


def seed_range(spec):
    lo, hi = (int(v) for v in spec.split("-"))
    return range(lo, hi + 1)


def job(a, exp_name, seed, results_dir, args, script=None, out=None):
    """``(output path, command)`` of one run."""
    results_dir = Path(results_dir)
    out = Path(out) if out is not None else results_dir / f"{seed:02d}_{exp_name}.pkl"
    cmd = [sys.executable, "-u", str(script or REPO / "train.py"),
           "--exp_name", exp_name, "--seed", str(seed),
           "--results_dir", str(results_dir), "--device", a.device]
    return out, cmd + [str(v) for v in args] + list(a.extra)


def pending(a, jobs):
    """Apply ``--shard`` and drop finished jobs unless ``--force``."""
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        jobs = jobs[i::n]
    if not a.force:
        jobs = [(o, c) for o, c in jobs if not Path(o).exists()]
    return jobs


def dispatch(name, jobs, a):
    """Run or print ``jobs`` according to the flags."""
    if a.dry_run:
        for _, cmd in jobs:
            print(shlex.join(str(c) for c in cmd))
        print(f"# {len(jobs)} run(s)", file=sys.stderr)
        return
    if not jobs:
        print(f"[{name}] nothing pending", flush=True)
        return
    print(f"[{name}] {len(jobs)} run(s) pending on {a.device}", flush=True)
    for k, (out, cmd) in enumerate(jobs, 1):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        print(f"\n=== [{name} {k}/{len(jobs)}] {Path(out).name} ===", flush=True)
        subprocess.run([str(c) for c in cmd], check=True)


def launch(name, configs, description="", seeds="0-9", results_dir=None, base=None):
    """Run every (configuration, seed) of one experiment, skipping what is done.

    ``base``: flags prepended to every configuration (a configuration's own
    flags come after, so they override a repeated flag).  Jobs are ordered
    seed-major, so interrupting the run leaves the seed counts balanced across
    configurations rather than all seeds of the first few and none of the rest.
    """
    a = parse_args(description, seeds=seeds)
    if base:
        configs = [(cfg, list(base) + list(args)) for cfg, args in configs]
    check(configs)
    results_dir = Path(a.results_dir or results_dir or RESULTS / name)
    jobs = [job(a, cfg, seed, results_dir, args)
            for seed in seed_range(a.seeds) for cfg, args in configs]
    jobs = pending(a, jobs)
    print(f"[{name}] {len(configs)} configuration(s) x seeds {a.seeds} "
          f"-> {len(jobs)} run(s) pending, results in {results_dir}", flush=True)
    dispatch(name, jobs, a)
