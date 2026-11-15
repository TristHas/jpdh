#!/usr/bin/env python3
"""Stage 1 of the two-stage recipe: the runoff module trained as a lumped
model, one checkpoint per (seed, fold).

For each fold the *same* ``diffhydro.Runoff`` as ``train.py``'s (2x256 LSTM,
softplus, same dynamic and static inputs, same normalisation) is trained on
the fold's training pool: inputs = the area-weighted mean over each gauge's
upstream catchments of the forcing and of the static descriptors, target =
the gauge's observed discharge, output converted with the total upstream area
exactly as ``RRModel.forward`` converts a catchment's runoff
(``mm_to_m3s(runoff_mm, area, 24)``) and compared with the y-normalised target
(``src/lumped.py``).  Loss, optimizer, learning rate, epochs, windows, cosine
schedule and SWA are ``train.py``'s.  The fold's val gauges (training-pool
gauges of the held-out river basins) are monitored, never selected on.

Output ``results/training_recipe/lumped/{seed:02d}_lumped_fold{fold}.pt``: the
runoff-module state dict (keys as in ``RRModel.runoff_model``, loadable with
``train.py --runoff_init``), every reported variant's weights, the
normalisation constants, the fold's training/val ids and per-gauge NSE
(training fit and val), the config and the library states.

    python experiments/train_lumped.py --seed 0 --device cuda:0
    python experiments/train_lumped.py --seed 0 --max_folds 1 --n_epoch 1 --n_iter 1   # smoke

``experiments/training_recipe.py --arms lumped`` launches the ten seeds.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402
import torch          # noqa: E402

from config import (DEFAULT_DYNAMIC_KEYS, DEFAULT_STATIC_RUNOFF_KEYS,   # noqa: E402
                    data_root)
from data import (data_loading, default_splits_path, read_splits_file,   # noqa: E402
                  time_blocks_path, load_time_blocks, block_mask, time_index,
                  mask_targets)
from lumped import (build_lumped_inputs, LumpedRunoffModel, LumpedModule,  # noqa: E402
                    lumped_dataset, runoff_state)
from model import mean_specific_runoff, init_output_bias                 # noqa: E402
from provenance import run_provenance                                    # noqa: E402
from training import VARIANTS, block_scores                              # noqa: E402

DEFAULT_RESULTS_DIR = REPO / "results" / "training_recipe" / "lumped"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp_name", default="lumped")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data_root", default=None,
                   help="dataset directory (default: $JPDH_DATA_ROOT or ./data)")
    p.add_argument("--dynamic_vars", default=",".join(DEFAULT_DYNAMIC_KEYS))
    p.add_argument("--static_runoff_vars", default=",".join(DEFAULT_STATIC_RUNOFF_KEYS))
    p.add_argument("--splits_file", default=None)
    p.add_argument("--time_blocks_file", default=None)
    p.add_argument("--holdout_block", default="none",
                   help="as train.py: withhold a temporal block from the "
                        "training targets (for a two-stage space-time run)")

    p.add_argument("--init_output_bias", choices=["none", "mean_runoff"], default="mean_runoff",
                   help="as train.py: start the output at the pool's mean specific runoff")
    p.add_argument("--n_epoch", type=int, default=120)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--inference_batch_size", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=100)
    p.add_argument("--init_window", type=int, default=365)
    p.add_argument("--runoff_hidden_size", type=int, default=256)
    p.add_argument("--runoff_num_layers", type=int, default=2)
    p.add_argument("--runoff_lr", type=float, default=1e-3)
    p.add_argument("--runoff_wd", type=float, default=0.0)
    p.add_argument("--scheduler", choices=["step", "cosine", "none"], default="cosine")
    p.add_argument("--scheduler_step_size", type=int, default=10000)
    p.add_argument("--scheduler_gamma", type=float, default=0.3)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--nse_eps", type=float, default=0.0)
    p.add_argument("--station_loss_clip", type=float, default=5.0)
    p.add_argument("--swa_start_frac", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_folds", type=int, default=None)
    p.add_argument("--folds", default=None, help="comma-separated fold indices")
    p.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--variants", default="swa",
                   help="weight variants to evaluate and store; swa (or else the "
                        "first listed) is the one train.py --runoff_init loads")
    p.add_argument("--force", action="store_true", help="re-train existing folds")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    dynamic_keys = [k.strip() for k in args.dynamic_vars.split(",")]
    static_runoff_keys = [k.strip() for k in args.static_runoff_vars.split(",")]
    variants = tuple(v.strip() for v in args.variants.split(","))
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown weight variant(s) {unknown}")
    main_variant = "swa" if "swa" in variants else variants[0]
    runoff_params = {"hidden_size": args.runoff_hidden_size,
                     "num_layers": args.runoff_num_layers}

    root = Path(args.data_root) if args.data_root else data_root()
    splits_path = Path(args.splits_file) if args.splits_file else default_splits_path(root)
    splits_meta = read_splits_file(splits_path)
    holdout = None
    if args.holdout_block != "none":
        tb = time_blocks_path(splits_path, splits_meta, args.time_blocks_file)
        if tb is None or not Path(tb).exists():
            raise SystemExit("--holdout_block: no time_blocks.json found")
        blocks = load_time_blocks(tb)
        holdout = dict(name=args.holdout_block, bounds=blocks[args.holdout_block],
                       file=str(Path(tb).resolve()))

    config = dict(vars(args), dynamic_keys=dynamic_keys,
                  static_runoff_keys=static_runoff_keys,
                  runoff_params=runoff_params, main_variant=main_variant,
                  data_root=str(root.resolve()),
                  splits_file=splits_meta["_path"], splits_sha256=splits_meta["_sha256"],
                  holdout_block_bounds=([str(b.date()) for b in holdout["bounds"]]
                                        if holdout else None),
                  time_blocks_file=holdout["file"] if holdout else None,
                  stage="lumped")
    config.update(run_provenance())
    print(f"[{args.exp_name}] Config: {config}", flush=True)

    (g, x, y, x_mean, x_std, y_std, runoff_statics, _routing_statics,
     _channel_length, area, splits) = data_loading(
        dynamic_keys=dynamic_keys, static_runoff_keys=static_runoff_keys,
        root=root, splits_file=splits_path)
    config["y_std"] = float(y_std.values)
    print(f"[{args.exp_name}] y_std = {config['y_std']:.4f}", flush=True)
    norm = dict(x_mean=x_mean.to_pandas(), x_std=x_std.to_pandas(),
                y_std=config["y_std"], dynamic_var=list(x_mean.coords["variable"]),
                static_var=list(runoff_statics.coords["variable"]))

    y_train, train_days, holdout_mask = y, None, None
    if holdout is not None:
        holdout_mask = block_mask(time_index(y), holdout["bounds"])
        y_train, train_days = mask_targets(y, holdout_mask), ~holdout_mask

    if args.max_folds is not None:
        splits = splits[:args.max_folds]
    fold_indices = sorted({int(i) for i in args.folds.split(",")}) \
        if args.folds is not None else None
    n_dyn, n_stat = len(x["variable"]), len(runoff_statics["variable"])
    seed = args.seed

    for fold_idx, (tr_nodes, val_nodes, _te_nodes) in enumerate(splits):
        if fold_indices is not None and fold_idx not in fold_indices:
            continue
        out_path = results_dir / f"{seed:02d}_{args.exp_name}_fold{fold_idx}.pt"
        if out_path.exists() and not args.force:
            print(f"[{args.exp_name}] fold {fold_idx}: {out_path} exists, skipping",
                  flush=True)
            continue
        t0 = time.time()
        torch.manual_seed(seed * 1000 + fold_idx)
        np.random.seed(seed * 1000 + fold_idx)
        tr_nodes, val_nodes = sorted(tr_nodes), sorted(val_nodes)

        x_l, s_l, a_l = build_lumped_inputs(g, x, runoff_statics, area,
                                            tr_nodes + val_nodes)
        ds_tr = lumped_dataset(x_l, s_l, a_l, y_train, tr_nodes, args.init_window,
                               args.pred_len, train_days=train_days)
        ds_val = lumped_dataset(x_l, s_l, a_l, y, val_nodes, args.init_window,
                                args.pred_len)
        # the training gauges' full, unmasked series: the "test" slot of the
        # module reports the training fit per gauge
        ds_fit = lumped_dataset(x_l, s_l, a_l, y, tr_nodes, args.init_window,
                                args.pred_len)
        print(f"[{args.exp_name}] fold {fold_idx}: {len(tr_nodes)} training gauges, "
              f"{len(val_nodes)} val gauges, {len(ds_tr)} windows, lumped areas "
              f"{float(a_l.min()):.0f}-{float(a_l.max()):.0f} km2", flush=True)

        model = LumpedRunoffModel(n_dyn + n_stat, runoff_params).to(device)
        if args.init_output_bias == "mean_runoff":
            target = mean_specific_runoff(y_train, g, area, tr_nodes)
            init_output_bias(model.runoff_model, target)
            print(f"  [fold {fold_idx}] runoff output bias initialised at {target:.4g}",
                  flush=True)
        module = LumpedModule(
            model, ds_tr, ds_val, ds_fit,
            scheduler_type=args.scheduler, n_epoch=args.n_epoch,
            nse_eps=args.nse_eps, station_loss_clip=args.station_loss_clip,
            swa_start_frac=args.swa_start_frac, batch_size=args.batch_size,
            inference_batch_size=args.inference_batch_size, device=device,
            clip_grad_norm=args.clip_grad_norm,
            routing_lr=0.0, routing_wd=0.0,
            runoff_lr=args.runoff_lr, runoff_wd=args.runoff_wd,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_gamma=args.scheduler_gamma)
        tr_losses, val_curve, best_epoch = module.train(
            n_epoch=args.n_epoch, n_iter=args.n_iter, track_val="best" in variants)

        nse_train, nse_val, extra = {}, {}, {}
        for variant in variants:
            ev = module.eval_variant(variant, device=device)
            nse_train[variant], nse_val[variant] = ev["nse_te"], ev["nse_val"]
            if holdout is not None:
                extra[variant] = {k: v for k, v in
                                  block_scores(ev, holdout["bounds"]).items()}
            print(f"  [fold {fold_idx}] {variant}: train fit med="
                  f"{float(np.nanmedian(ev['nse_te'].values)):.4f} "
                  f"val med={float(np.nanmedian(ev['nse_val'].values)):.4f}",
                  flush=True)

        payload = dict(
            state_dict=runoff_state(module.weight_variants[main_variant]),
            weights={v: runoff_state(module.weight_variants[v]) for v in variants},
            main_variant=main_variant, fold=fold_idx, seed=seed,
            train_nodes=tr_nodes, val_nodes=val_nodes,
            lumped_area_km2=pd.Series(a_l.numpy(), index=tr_nodes + val_nodes),
            nse_train=nse_train, nse_val=nse_val, block_scores=extra,
            nse_train_median=float(np.nanmedian(nse_train[main_variant].values)),
            nse_val_median=float(np.nanmedian(nse_val[main_variant].values)),
            tr_losses=tr_losses, val_curve=val_curve, best_epoch=best_epoch,
            norm=norm, config=config, wall_time_s=time.time() - t0,
        )
        torch.save(payload, out_path)
        print(f"[{args.exp_name}] fold {fold_idx} -> {out_path} "
              f"({payload['wall_time_s'] / 60:.1f} min)", flush=True)
        model = module = ds_tr = ds_val = ds_fit = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
