"""Train and evaluate the coupled runoff-routing model over the PUB folds.

    python train.py --exp_name default                   # the default configuration, 10 folds
    python train.py --exp_name smoke --max_folds 1 --n_epoch 2 --n_iter 2   # quick check

Every default is the default configuration of the paper (Table 2): in-situ
precipitation and MSM temperature in 4-h bins, a 2 x 256 LSTM runoff module,
linear-storage routing whose time constant comes from an MLP on the reach's
channel length, end-to-end training on the per-station NSE loss, SWA weights.
One pickle per seed is written to ``--results_dir`` with the per-gauge scores,
optionally the simulated series (``--save_series``) and the weights
(``--save_weights``; ``scripts/export_weights.py`` turns them into model files).
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

from config import (
    DEFAULT_DYNAMIC_KEYS, DEFAULT_IRF, DEFAULT_ROUTING_INIT, DEFAULT_ROUTING_LR,
    DEFAULT_ROUTING_STATIC_VAR, DEFAULT_STATIC_RUNOFF_KEYS, KERNEL_BOUNDS, data_root,
)
from data import (data_loading, default_splits_path, read_splits_file,
                  time_blocks_path, load_time_blocks, block_mask, time_index)
from model import IRF_CHOICES, parse_routing_bounds
from provenance import run_provenance
from training import run_experiments, VARIANTS

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data_root", default=None,
                   help="dataset directory (default: $JPDH_DATA_ROOT or ./data)")

    # inputs
    p.add_argument("--dynamic_vars", default=",".join(DEFAULT_DYNAMIC_KEYS),
                   help="comma-separated keys of config.DYNAMIC_VAR_DICT")
    p.add_argument("--static_runoff_vars", default=",".join(DEFAULT_STATIC_RUNOFF_KEYS),
                   help="comma-separated keys of config.STATIC_VAR_DICT")
    p.add_argument("--routing_static_vars", default=",".join(DEFAULT_ROUTING_STATIC_VAR),
                   help="descriptors of the routing-parameter MLP: channel_length_km, "
                        "log_channel_length_km or columns of statics/routing_statics.parquet "
                        "(e.g. log_upa, log_slope, sinuosity)")

    # routing
    p.add_argument("--irf_fn", default=DEFAULT_IRF, choices=IRF_CHOICES,
                   help="DiffRoute routing kernel, or 'instantaneous' for the no-routing "
                        "arm (exact upstream sum, no routing parameters)")
    p.add_argument("--routing_bounds", default=None,
                   help="parameter bounds, e.g. 'tau=0.01,9.9' (hourly units of DiffHydro's "
                        "PARAMS_BOUNDS; several slots separated by ';'). Default: "
                        "config.KERNEL_BOUNDS of the kernel; 'none' = DiffHydro's own")
    p.add_argument("--routing_init", choices=["low", "mid", "high", "none"],
                   default=DEFAULT_ROUTING_INIT,
                   help="initial value of every routing parameter as a fraction of its range "
                        "through the sigmoid (low = 4.7 %%, mid = 50 %%, high = 95.3 %%); "
                        "'none' = DiffHydro's default")
    p.add_argument("--route_src_reach", action="store_true",
                   help="also route each catchment's own runoff through that catchment's "
                        "own reach. Off by default: the runoff module already maps forcing "
                        "to the catchment outlet, so its own reach has been traversed.")

    # evaluation protocol: splits and temporal blocks
    p.add_argument("--splits_file", default=None,
                   help="folds (default: the dataset's splits/pub_folds_10.json)")
    p.add_argument("--time_blocks_file", default=None,
                   help="time_blocks.json; default: the splits file's time_blocks_file key, "
                        "else the file next to it")
    p.add_argument("--holdout_block", default="none",
                   help="temporal block withheld from the training targets (B1, B2, B3: "
                        "PUB-T setting). Training windows are sampled inside the other "
                        "blocks, evaluation runs the full continuous series; requires "
                        "--save_series since the block is scored from the saved series.")

    # two-stage recipe
    p.add_argument("--runoff_init", default=None,
                   help="runoff-module checkpoint template with {seed} (or {seed:02d}) and "
                        "{fold}: an experiments/train_lumped.py .pt, or a train.py .pkl "
                        "produced with --save_weights")
    p.add_argument("--freeze_runoff", action="store_true",
                   help="keep the initialised runoff module fixed: only the routing "
                        "parameters train")
    p.add_argument("--init_output_bias", choices=["none", "mean_runoff"], default="mean_runoff",
                   help="'mean_runoff': start the runoff module's output at the training "
                        "pool's mean specific runoff, so stations begin inside the per-station "
                        "loss clip; 'none': DiffHydro's initialisation. Ignored with --runoff_init.")

    # schedule
    p.add_argument("--n_epoch", type=int, default=120)
    p.add_argument("--n_iter", type=int, default=20, help="training windows per epoch")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--inference_batch_size", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=100, help="scored days per window")
    p.add_argument("--init_window", type=int, default=365, help="warm-up days per window")

    # architecture
    p.add_argument("--runoff_hidden_size", type=int, default=256)
    p.add_argument("--runoff_num_layers", type=int, default=2)

    # optimisation
    p.add_argument("--runoff_lr", type=float, default=1e-3)
    p.add_argument("--runoff_wd", type=float, default=0.0)
    p.add_argument("--routing_lr", type=float, default=DEFAULT_ROUTING_LR)
    p.add_argument("--routing_wd", type=float, default=0.0)
    p.add_argument("--scheduler", choices=["step", "cosine", "none"], default="cosine")
    p.add_argument("--scheduler_step_size", type=int, default=10000)
    p.add_argument("--scheduler_gamma", type=float, default=0.3)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    # loss
    p.add_argument("--nse_eps", type=float, default=0.0)
    p.add_argument("--station_loss_clip", type=float, default=5.0)

    # weights and outputs
    p.add_argument("--swa_start_frac", type=float, default=0.75)
    p.add_argument("--variants", default="swa",
                   help="comma-separated weight variants to evaluate and report among "
                        f"{','.join(VARIANTS)} (default swa, the reported one). 'best' "
                        "adds a validation pass every epoch and must not be reported: "
                        "under PUB the validation gauges lie in the held-out river basins.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_repeats", type=int, default=1, help="seeds seed .. seed + n_repeats - 1")
    p.add_argument("--max_folds", type=int, default=None)
    p.add_argument("--folds", default=None,
                   help="comma-separated fold indices, e.g. '0,3,7'. Folds are independent and "
                        "seeded by their index, so running them on separate GPUs and merging "
                        "with scripts/merge_folds.py gives the same result as one run.")
    p.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--save_series", action="store_true",
                   help="store the simulated and observed val/test series")
    p.add_argument("--save_weights", action="store_true",
                   help="store each reported variant's state dict (~3 MB per fold)")
    return p.parse_args(argv)


def resolve_routing(args):
    """``(bounds dict or None, init or None, bounds string)`` of the run."""
    spec = args.routing_bounds
    if spec is None:
        spec = KERNEL_BOUNDS.get(args.irf_fn, "")
    if spec == "none":
        spec = ""
    init = None if args.routing_init == "none" else args.routing_init
    return parse_routing_bounds(spec) or None, init, spec


def resolve_holdout(args, splits_path, splits_meta):
    """The held-out block of a run: ``None`` or a dict with name, bounds,
    the blocks file and every block's dates (all stored in the config)."""
    if args.holdout_block == "none":
        return None
    if not args.save_series:
        raise SystemExit("--holdout_block requires --save_series: the held-out "
                         "block is scored from the saved continuous series")
    tb = time_blocks_path(splits_path, splits_meta, args.time_blocks_file)
    if tb is None or not Path(tb).exists():
        raise SystemExit(f"--holdout_block {args.holdout_block}: no time_blocks.json "
                         f"found (looked at --time_blocks_file, the splits file's "
                         f"time_blocks_file key and next to {splits_path})")
    blocks = load_time_blocks(tb)
    if args.holdout_block not in blocks:
        raise SystemExit(f"--holdout_block {args.holdout_block!r} not in {tb}: "
                         f"{sorted(blocks)}")
    return dict(name=args.holdout_block, bounds=blocks[args.holdout_block],
                file=str(Path(tb).resolve()),
                blocks={k: [str(a.date()), str(b.date())] for k, (a, b) in blocks.items()})


def main(argv=None):
    args = parse_args(argv)
    t_run = time.time()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root) if args.data_root else data_root()

    dynamic_keys = [k.strip() for k in args.dynamic_vars.split(",")]
    static_runoff_keys = [k.strip() for k in args.static_runoff_vars.split(",")]
    routing_features = [k.strip() for k in args.routing_static_vars.split(",")]

    if args.freeze_runoff and not args.runoff_init:
        raise SystemExit("--freeze_runoff needs --runoff_init (nothing to freeze)")
    routing_bounds, routing_init, bounds_spec = resolve_routing(args)

    variants = tuple(v.strip() for v in args.variants.split(","))
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown weight variant(s) {unknown}. "
                         f"Known: {list(VARIANTS)}")

    runoff_params = {
        "hidden_size": args.runoff_hidden_size,
        "num_layers":  args.runoff_num_layers,
    }

    # ---- protocol inputs and provenance ----
    splits_path = Path(args.splits_file) if args.splits_file else default_splits_path(root)
    splits_meta = read_splits_file(splits_path)
    holdout = resolve_holdout(args, splits_path, splits_meta)

    config = dict(vars(args))
    config["dynamic_keys"] = dynamic_keys
    config["static_runoff_keys"] = static_runoff_keys
    config["routing_features"] = routing_features
    config["routing_bounds"] = bounds_spec
    config["routing_init"] = routing_init
    config.update(run_provenance())
    config.update(
        data_root=str(root.resolve()),
        splits_file=splits_meta["_path"],
        splits_sha256=splits_meta["_sha256"],
        holdout_block=args.holdout_block,
        holdout_block_bounds=([str(b.date()) for b in holdout["bounds"]]
                              if holdout else None),
        time_blocks_file=holdout["file"] if holdout else None,
        time_blocks=holdout["blocks"] if holdout else None,
    )

    print(f"[{args.exp_name}] Config: {config}", flush=True)
    print(f"[{args.exp_name}] Loading data from {root} ...", flush=True)

    (g, x, y, x_mean, x_std, y_std,
     runoff_statics, routing_statics,
     channel_length, area, splits) = data_loading(
        dynamic_keys=dynamic_keys,
        static_runoff_keys=static_runoff_keys,
        routing_static_var=routing_features,
        root=root,
        splits_file=splits_path,
    )
    config["y_std"] = float(y_std.values)
    config["n_norm_gauges"] = len(splits_meta["gauges_with_discharge"])
    print(f"[{args.exp_name}] y_std = {config['y_std']:.4f} m3/s over "
          f"{config['n_norm_gauges']} gauges", flush=True)

    holdout_mask = None
    if holdout is not None:
        holdout_mask = block_mask(time_index(y), holdout["bounds"])
        print(f"[{args.exp_name}] holding out block {holdout['name']} "
              f"{config['holdout_block_bounds']}: {int(holdout_mask.sum())} days",
              flush=True)

    if args.max_folds is not None:
        splits = splits[:args.max_folds]
        print(f"[{args.exp_name}] Limiting to {args.max_folds} fold(s).",
              flush=True)

    fold_indices = None
    if args.folds is not None:
        fold_indices = sorted({int(i) for i in args.folds.split(",")})
        bad = [i for i in fold_indices if not 0 <= i < len(splits)]
        if bad:
            raise ValueError(f"fold indices out of range 0..{len(splits)-1}: {bad}")
        print(f"[{args.exp_name}] Running folds {fold_indices} of "
              f"{len(splits)}.", flush=True)

    common = dict(
        device=args.device,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        n_epoch=args.n_epoch,
        n_iter=args.n_iter,
        init_window=args.init_window,
        pred_len=args.pred_len,
        runoff_params=runoff_params,
        runoff_lr=args.runoff_lr,
        runoff_wd=args.runoff_wd,
        routing_lr=args.routing_lr,
        routing_wd=args.routing_wd,
        scheduler_type=args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        clip_grad_norm=args.clip_grad_norm,
        nse_eps=args.nse_eps,
        station_loss_clip=args.station_loss_clip,
        swa_start_frac=args.swa_start_frac,
        route_src_reach=args.route_src_reach,
        save_series=args.save_series,
        save_weights=args.save_weights,
        variants=variants,
        irf_fn=args.irf_fn,
        holdout_mask=holdout_mask,
        holdout_bounds=holdout["bounds"] if holdout else None,
        runoff_init=args.runoff_init,
        freeze_runoff=args.freeze_runoff,
        runoff_init_check=(dynamic_keys, static_runoff_keys),
        init_output_bias_mode=args.init_output_bias,
        routing_bounds=routing_bounds,
        routing_init=routing_init,
    )

    for repeat_idx in range(args.n_repeats):
        seed = args.seed + repeat_idx
        tag = f"{seed:02d}_{args.exp_name}"
        if fold_indices is not None:
            tag += "_folds" + "-".join(str(i) for i in fold_indices)
        out_path = results_dir / f"{tag}.pkl"

        print(f"[{tag}] Running on {args.device} (seed {seed}) ...", flush=True)
        t_seed = time.time()
        fold_results = run_experiments(
            g, x, y,
            runoff_statics, routing_statics,
            channel_length, area,
            splits, routing_features,
            seed=seed, fold_indices=fold_indices, **common,
        )

        summary = {}
        for variant in variants:
            nse_val = pd.concat([fr[variant]["nse_val"] for fr in fold_results])
            nse_te = pd.concat([fr[variant]["nse_te"] for fr in fold_results])
            summary[variant] = dict(
                val_median=float(np.nanmedian(nse_val.values)),
                val_mean=float(np.nanmean(nse_val.values)),
                te_median=float(np.nanmedian(nse_te.values)),
                te_mean=float(np.nanmean(nse_te.values)),
            )
            line = (f"[{tag}] {variant}: "
                    f"val med={summary[variant]['val_median']:.4f} "
                    f"mean={summary[variant]['val_mean']:.4f} | "
                    f"te med={summary[variant]['te_median']:.4f} "
                    f"mean={summary[variant]['te_mean']:.4f}")
            if holdout is not None:
                for key in ("nse_te_holdout", "nse_te_trainblocks",
                            "nse_val_holdout", "nse_val_trainblocks"):
                    s = pd.concat([fr[variant][key] for fr in fold_results])
                    summary[variant][key.replace("nse_", "") + "_median"] = \
                        float(np.nanmedian(s.values))
                line += (f" | te held-out block med="
                         f"{summary[variant]['te_holdout_median']:.4f} "
                         f"training blocks med="
                         f"{summary[variant]['te_trainblocks_median']:.4f}")
            print(line, flush=True)

        config = dict(config, seed=seed, wall_time_s=time.time() - t_seed,
                      fold_wall_time_s={fr["fold"]: fr["wall_time_s"]
                                        for fr in fold_results},
                      finished_at=datetime.datetime.now().astimezone()
                      .isoformat(timespec="seconds"))
        pd.to_pickle(
            dict(config=config, summary=summary, fold_results=fold_results,
                 x_mean=x_mean, x_std=x_std, y_std=y_std),
            out_path,
        )
        print(f"[{tag}] Saved to {out_path} ({(time.time() - t_run) / 60:.1f} min "
              f"since start)", flush=True)


if __name__ == "__main__":
    main()
