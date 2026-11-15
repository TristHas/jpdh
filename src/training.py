from __future__ import annotations
import torch
import torch.nn as nn

def clipped_nse_loss(o, y, var, nse_eps: float = 0.0,
                     station_loss_clip: float | None = None) -> torch.Tensor:
    """Mean per-station NSE-style loss.

    ``o``, ``y`` are DataTensors shaped (batch, spatial, time); ``var`` holds the
    per-station target variance. The squared error is summed over (batch, time)
    and normalised per station, then divided by the variance to give a per-station
    NSE complement which is averaged over stations.

    Parameters
    ----------
    nse_eps:
        Kratzert-style eps damping of the denominator, ``(std + eps)**2``. With
        ``0.0`` the plain variance is used (standard NSE).
    station_loss_clip:
        Upper bound on the per-station loss before averaging. Caps the gradient
        contribution of badly-fit / blow-up stations (outlier thresholding).
        ``None`` disables clipping.
    """
    y = y.values
    o = o.values
    var = var.values.squeeze()

    valid = ~y.isnan()
    safe_o = torch.where(valid, o, torch.zeros_like(o))
    safe_y = torch.where(valid, y, torch.zeros_like(y))

    mse = ((safe_o - safe_y) ** 2).sum((0, -1)) / valid.sum((0, -1)).clip(1)

    if nse_eps > 0:
        denom = (var.sqrt() + nse_eps) ** 2
    else:
        denom = var
    
    per_station = mse / denom
    # a station whose target variance is not finite (no valid target at all,
    # e.g. a record confined to a held-out temporal block) contributes zero,
    # as one without a valid target in the window already does
    per_station = torch.where(torch.isfinite(per_station), per_station,
                              torch.zeros_like(per_station))

    if station_loss_clip is not None:
        per_station = per_station.clamp(max=station_loss_clip)
    
    return per_station.mean()


def state_to_cpu(model: nn.Module) -> dict:
    """
        Detached CPU copy of a model's state dict (for checkpoint snapshots).
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def swa_update(avg: dict | None, state: dict, n: int) -> dict:
    """
        Running average of state dicts (floats averaged, others copied).
    """
    if avg is None:
        return {k: v.clone() for k, v in state.items()}
    for k, v in state.items():
        if v.is_floating_point():
            avg[k] += (v - avg[k]) / (n + 1)
        else:
            avg[k] = v.clone()
    return avg


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------

def build_scheduler(opt, scheduler_type: str, n_epoch: int,
                    step_size: int | None = None, gamma: float = 0.1):
    """
    Construct a per-epoch LR scheduler.

    ``"cosine"`` -> CosineAnnealingLR(T_max=n_epoch); ``"step"`` -> StepLR;
    ``"none"``/``None`` -> no scheduler.
    """
    if scheduler_type in ("none", None):
        return None
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epoch, eta_min=0.0)
    if scheduler_type == "step":
        if step_size is None:
            raise ValueError("step_size is required for scheduler_type='step'")
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=step_size, gamma=gamma)
    raise ValueError(f"unknown scheduler_type {scheduler_type!r}")


# ---------------------------------------------------------------------------
# Training module and experiment driver
# ---------------------------------------------------------------------------

import copy
import time
import numpy as np
import pandas as pd
import diffhydro.pipelines as dhp
import xtensor as xt
from diffhydro.pipelines.utils import collate_fn, PARAMS_BOUNDS

from data import init_split_dataset, mask_targets, block_mask
from model import (JPDHRRModel, build_param_model,
                   override_router_buffers, resolve_runoff_init, load_runoff_state,
                   check_runoff_init_config, mean_specific_runoff,
                   init_output_bias)

#: the routing operator's time base: a daily runoff step (temp_res_h), resolved
#: hourly inside the kernel (dt), truncated at max_delay days
TEMP_RES_H, MAX_DELAY, DT = 24, 30, 1 / 24


def build_model(irf_fn, routing_features, n_dyn, n_stat, runoff_params, device,
                routing_bounds=None, routing_init=None):
    """The coupled runoff-routing model, plus its routing-parameter module.

    ``routing_bounds`` / ``routing_init`` (``model.override_router_buffers``)
    replace DiffHydro's default bounds and initial value of the kernel's
    parameters.  Used both for training and for rebuilding a saved model
    (``weights.load_model``), so a saved state dict always loads back.
    """
    param_model = build_param_model(irf_fn, routing_features)
    model = JPDHRRModel(param_model, input_size=n_dyn + n_stat,
                        max_delay=MAX_DELAY, dt=DT, irf_name=irf_fn,
                        temp_res_h=TEMP_RES_H, runoff_params=runoff_params).to(device)
    if irf_fn != "instantaneous" and (routing_bounds or routing_init is not None):
        override_router_buffers(model.routing_model, irf_fn, TEMP_RES_H,
                                bounds=routing_bounds, init=routing_init)
    return model, param_model


class RRModule(dhp.RRModule):
    def __init__(self, model, tr_ds, val_ds, te_ds,
                 scheduler_type="step",
                 n_epoch=40,
                 nse_eps=0.0,
                 station_loss_clip=None,
                 swa_start_frac=0.75,
                 **kwargs):
        super().__init__(model, tr_ds, val_ds, te_ds, **kwargs)
        self.nse_eps = nse_eps
        self.station_loss_clip = station_loss_clip
        self.swa_start_frac = swa_start_frac

        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=n_epoch, eta_min=0.0)
        elif scheduler_type == "none":
            self.scheduler = None
        # "step": keep the StepLR installed by RRModule.init_optimizer

    def init_optimizer(self,
                       routing_lr=10**-4,
                       routing_wd=10**-3,
                       runoff_lr=.005,
                       runoff_wd=.001,
                       scheduler_step_size=None,
                       scheduler_gamma=.1):
        """Two AdamW groups (runoff / routing) over the *trainable* parameters.

        Same groups and learning rates as ``dhp.RRModule.init_optimizer``,
        except that parameters with ``requires_grad=False`` are left out and a
        group with nothing left is dropped: a frozen runoff module (two-stage
        arms, ``JPDHRRModel.freeze_runoff``) leaves only the routing group,
        which keeps ``routing_lr``; the instantaneous router has no parameters
        at all; the lumped model has no routing module.  A frozen module also
        no longer counts in the gradient-norm clip, since it has no gradients.
        """
        groups = []
        for name, module, lr, wd in (
                ("runoff", getattr(self.model, "runoff_model", None), runoff_lr, runoff_wd),
                ("routing", getattr(self.model, "routing_model", None), routing_lr, routing_wd)):
            params = [p for p in module.parameters() if p.requires_grad] \
                if module is not None else []
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": wd,
                               "name": name})
        if not groups:
            raise ValueError("nothing to optimize: every parameter is frozen "
                             "(--freeze_runoff with a parameter-less routing scheme?)")
        self.opt = torch.optim.AdamW(groups)
        if scheduler_step_size is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.opt, step_size=scheduler_step_size, gamma=scheduler_gamma)
        else:
            self.scheduler = None

    def loss_fn(self, o, y, var):
        return clipped_nse_loss(o, y, var, nse_eps=self.nse_eps,
                                station_loss_clip=self.station_loss_clip)

    def train(self, n_epoch, n_iter=None, device=None, track_val=True):
        """Train while tracking best-val-epoch and SWA weight variants.

        With ``track_val=False`` the per-epoch val extraction is skipped
        (no val set, or fixed-schedule protocols where val-based selection
        would leak); only the 'final' and 'swa' variants are produced.

        Returns (tr_losses, val_curve, best_epoch) where val_curve is a list
        of (median, mean) val NSE per epoch.  After training,
        ``self.weight_variants`` maps variant name -> CPU state dict.
        """
        device = device or self.device
        self.model = self.model.to(device)

        tr_losses, val_curve = [], []
        best_median, best_epoch, best_state = -np.inf, -1, None
        swa_state, swa_n = None, 0
        swa_start = int(self.swa_start_frac * n_epoch)

        for epoch in range(n_epoch):
            tr_loss = self.train_epoch(device=device, n_iter=n_iter)
            tr_losses.append(tr_loss)

            if track_val:
                val_nse = self.valid_epoch(device=device)
                v = val_nse.values.cpu().numpy().ravel()
                med, mean = float(np.nanmedian(v)), float(np.nanmean(v))
                val_curve.append((med, mean))

                if med > best_median:
                    best_median, best_epoch = med, epoch
                    best_state = state_to_cpu(self.model)

            if epoch >= swa_start:
                swa_state = swa_update(swa_state, state_to_cpu(self.model),
                                       swa_n)
                swa_n += 1

            torch.cuda.empty_cache()

        self.weight_variants = {
            "final": state_to_cpu(self.model),
            "swa":   swa_state,
        }
        if best_state is not None:
            self.weight_variants["best"] = best_state
        self.best_epoch = best_epoch
        return tr_losses, val_curve, best_epoch

    def _extract_full_ts(self, ds, batch_size, device=None):
        """The continuous series over the *whole* evaluation period.

        ``dhp.BaseModule._extract_full_ts`` strides windows of ``pred_len``
        from the first start index and stops at the last full stride, so the
        final ``(T - init_len) mod pred_len`` days of the record (85 days of
        2020 with the paper's 365 + 100 windows) never appear in the series.
        That truncation would shorten block B3 and the released benchmark
        series, so one more window, ending on the last day, supplies the
        missing tail.
        """
        device = device or self.device
        y, o = super()._extract_full_ts(ds, batch_size, device)
        n_time = len(ds.y.coords["time"])
        covered = ds.init_len + y.shape[-1]
        missing = n_time - covered
        if missing <= 0:
            return y, o
        # a plain window slice, whatever the dataset's index mapping
        xw, yw = dhp.BaseDataset.__getitem__(ds, n_time - ds.total_len)
        xw, yw = collate_fn([(xw, yw)])
        with torch.no_grad():
            ow = self.run_model(ds, xw.to(device)).sel(spatial=yw.coords["spatial"])
        tail_coords = {"spatial": ds.y.coords["spatial"],
                       "time": ds.y.coords["time"][n_time - missing:]}

        def tail(values):
            vals = values.isel(time=slice(-missing, None)).values[0]
            return xt.DataTensor(vals.to(y.values.device), dims=("spatial", "time"),
                                 coords=tail_coords)
        return (xt.concat([y, tail(yw.to(device))], "time"),
                xt.concat([o, tail(ow)], "time"))

    def eval_variant(self, name, device=None):
        """Load a tracked weight variant and extract val/test NSE + series."""
        device = device or self.device
        self.model.load_state_dict(self.weight_variants[name])
        self.model = self.model.to(device)

        yval, oval = self.extract_val(device=device,
                                      batch_size=self.inference_batch_size)
        torch.cuda.empty_cache()
        yte, ote = self.extract_test(device=device, batch_size=1)
        torch.cuda.empty_cache()

        nse_val = 1 - (((yval - oval) ** 2).mean("time") / yval.var("time"))
        nse_te = 1 - (((yte - ote) ** 2).mean("time") / yte.var("time"))
        return dict(
            nse_val=nse_val.to_pandas(), nse_te=nse_te.to_pandas(),
            oval=oval.to_pandas().T, yval=yval.to_pandas().T,
            ote=ote.to_pandas().T, yte=yte.to_pandas().T,
        )


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------

VARIANTS = ("final", "best", "swa")


def series_nse(y: pd.DataFrame, o: pd.DataFrame, rows=None) -> pd.Series:
    """Per-column NSE of two (time, gauge) tables over ``rows`` (a boolean over
    time), NaN-aware and with the population variance -- the same statistic
    ``eval_variant`` reports over the full series."""
    if rows is not None:
        y, o = y[rows], o[rows]
    return 1 - ((y - o) ** 2).mean() / y.var(ddof=0)


def block_scores(ev: dict, bounds) -> dict:
    """The run-log sanity check of a block run: NSE inside the held-out block
    and inside the training blocks, from the extracted series of the fold's
    val (training gauges of the held-out river basins) and test (evaluation
    gauges) sets.  Full per-gauge scores: ``metrics.per_gauge_metrics``."""
    out = {}
    for tag in ("val", "te"):
        y, o = ev[f"y{tag}"], ev[f"o{tag}"]
        held = block_mask(y.index, bounds).numpy()
        out[f"nse_{tag}_holdout"] = series_nse(y, o, held)
        out[f"nse_{tag}_trainblocks"] = series_nse(y, o, ~held)
    return out


def run_experiments(
    g, x, y, runoff_statics, routing_statics,
    channel_length, area, splits, routing_features,
    device, batch_size, inference_batch_size,
    n_epoch, n_iter, init_window, pred_len,
    runoff_params,
    runoff_lr, runoff_wd, routing_lr, routing_wd,
    scheduler_type, scheduler_step_size, scheduler_gamma,
    clip_grad_norm,
    nse_eps=0.0, station_loss_clip=None, swa_start_frac=0.75,
    #: False = a catchment's own runoff is NOT routed through its own reach,
    #: because the runoff module already maps forcing to the catchment outlet.
    #: Upstream contributions to that outlet are still routed through it.
    route_src_reach=False,
    seed=0,
    save_series=False,
    save_weights=False,
    variants=VARIANTS,
    irf_fn="linear_storage",
    #: the router's parameter bounds ({slot: (min, max)}, hourly units) and
    #: initial value ("low" / "mid" / "high"); None = DiffHydro's defaults
    routing_bounds=None, routing_init=None,
    fold_indices=None,
    #: temporal holdout: boolean over the time axis, True on the days whose
    #: targets are withheld from training, plus the block's date bounds
    holdout_mask=None, holdout_bounds=None,
    #: two-stage recipe: a runoff-module checkpoint template and whether to
    #: keep it fixed; ``runoff_init_check`` = (dynamic_keys,
    #: static_runoff_keys) of this run, compared with the checkpoint's
    runoff_init=None, freeze_runoff=False, runoff_init_check=None,
    #: "mean_runoff": start the runoff module at the training pool's mean
    #: specific runoff (model.init_output_bias); "none": DiffHydro's initialisation
    init_output_bias_mode="mean_runoff",
):
    n_dyn = len(x["variable"])
    n_stat = len(runoff_statics["variable"])
    # 'best' is the only variant needing the per-epoch validation pass
    track_val = "best" in variants

    y_train, train_days = None, None
    if holdout_mask is not None:
        holdout_mask = torch.as_tensor(holdout_mask, dtype=torch.bool)
        y_train = mask_targets(y, holdout_mask)
        train_days = ~holdout_mask
        print(f"  [holdout] {int(holdout_mask.sum())} of {len(holdout_mask)} "
              f"days withheld from the training targets", flush=True)

    results = []
    for fold_idx, (tr_nodes, val_nodes, te_nodes) in enumerate(splits):
        if fold_indices is not None and fold_idx not in fold_indices:
            continue
        t_start = time.time()
        # Seeding is keyed on the *original* fold index, so running a subset of
        # folds (e.g. one per GPU) reproduces exactly what the full loop would
        # have produced for those folds.
        torch.manual_seed(seed * 1000 + fold_idx)
        np.random.seed(seed * 1000 + fold_idx)

        tr_nodes, val_nodes, te_nodes = map(list, [tr_nodes, val_nodes, te_nodes])
        if y_train is not None:
            # a training gauge whose record lies entirely inside the held-out
            # block has nothing to teach; keep it out of the training set (it
            # stays in val/test, where the full series is scored)
            has_target = torch.isfinite(y_train.sel(spatial=tr_nodes).values)\
                .any(dim=(0, -1)).cpu()
            dropped = [n for n, ok in zip(tr_nodes, has_target.tolist()) if not ok]
            if dropped:
                print(f"  [fold {fold_idx}] {len(dropped)} training gauge(s) without "
                      f"any target inside the training blocks left out: {dropped}",
                      flush=True)
                tr_nodes = [n for n in tr_nodes if n not in set(dropped)]
        tr_ds, val_ds, te_ds = init_split_dataset(
            g, x, y,
            runoff_statics, routing_statics,
            channel_length, area,
            tr_nodes, val_nodes, te_nodes,
            init_window, pred_len,
            irf_fn=irf_fn,
            route_src_reach=route_src_reach,
            y_train=y_train, train_days=train_days,
        )
        if holdout_mask is not None:
            # the temporal holdout's guarantee, asserted where the run log shows it
            assert torch.isnan(tr_ds.y.values[..., holdout_mask]).all(), \
                "held-out targets leaked into the training dataset"
            assert torch.isfinite(te_ds.y.values[..., holdout_mask]).any(), \
                "no evaluation target inside the held-out block"
            print(f"  [fold {fold_idx}] {len(tr_ds)} training windows with their "
                  f"prediction part inside the training blocks; held-out targets "
                  f"are NaN in the training set", flush=True)

        model, param_model = build_model(
            irf_fn, routing_features, n_dyn, n_stat, runoff_params, device,
            routing_bounds=routing_bounds, routing_init=routing_init)

        init_path = None
        if runoff_init:
            init_path = resolve_runoff_init(runoff_init, seed, fold_idx)
            state, meta = load_runoff_state(init_path, fold=fold_idx)
            if runoff_init_check is not None:
                check_runoff_init_config(meta, *runoff_init_check)
            model.runoff_model.load_state_dict(state)
            print(f"  [fold {fold_idx}] runoff module initialised from {init_path}",
                  flush=True)
        if init_output_bias_mode == "mean_runoff" and init_path is None:
            target = mean_specific_runoff(y_train if y_train is not None else y,
                                          g, area, tr_nodes, TEMP_RES_H)
            init_output_bias(model.runoff_model, target)
            print(f"  [fold {fold_idx}] runoff output bias initialised at "
                  f"{target:.4g} (normalised mm per step)", flush=True)
        if freeze_runoff:
            model.freeze_runoff()
            print(f"  [fold {fold_idx}] runoff module frozen; trainable parameters: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}",
                  flush=True)

        module = RRModule(
            model, tr_ds, val_ds, te_ds,
            scheduler_type=scheduler_type,
            n_epoch=n_epoch,
            nse_eps=nse_eps,
            station_loss_clip=station_loss_clip,
            swa_start_frac=swa_start_frac,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            device=device,
            clip_grad_norm=clip_grad_norm,
            routing_lr=routing_lr,
            routing_wd=routing_wd,
            runoff_lr=runoff_lr,
            runoff_wd=runoff_wd,
            scheduler_step_size=scheduler_step_size,
            scheduler_gamma=scheduler_gamma,
        )

        tr_losses, val_curve, best_epoch = module.train(n_epoch=n_epoch,
                                                        n_iter=n_iter,
                                                        track_val=track_val)

        fold_result = dict(
            fold=fold_idx,
            tr_losses=tr_losses,
            val_curve=val_curve,
            best_epoch=best_epoch,
            routing_param_names=(list(PARAMS_BOUNDS[irf_fn].columns)
                                 if irf_fn in PARAMS_BOUNDS else None),
            n_train_windows=len(tr_ds),
            runoff_init=str(init_path) if init_path is not None else None,
            init_output_bias=(init_output_bias_mode if init_path is None else "none"),
            optimizer_groups=[g_["name"] for g_ in module.opt.param_groups],
        )
        
        for variant in variants:
            ev = module.eval_variant(variant, device=device)
            if holdout_mask is not None:
                ev.update(block_scores(ev, holdout_bounds))
            if not save_series:
                ev = {k: v for k, v in ev.items() if k.startswith("nse")}
            if save_weights:
                ev["weights"] = module.weight_variants[variant]
            fold_result[variant] = ev
            med = float(np.nanmedian(ev["nse_te"].values))
            line = (f"  [fold {fold_idx}] {variant}: "
                    f"val med={float(np.nanmedian(ev['nse_val'].values)):.4f} "
                    f"te med={med:.4f}")
            if holdout_mask is not None:
                line += (f" | te held-out block med="
                         f"{float(np.nanmedian(ev['nse_te_holdout'].values)):.4f}"
                         f" training blocks med="
                         f"{float(np.nanmedian(ev['nse_te_trainblocks'].values)):.4f}")
            print(line, flush=True)

        fold_result["wall_time_s"] = time.time() - t_start
        print(f"  [fold {fold_idx}] wall time {fold_result['wall_time_s'] / 60:.1f} min",
              flush=True)
        results.append(fold_result)

        model = module = param_model = tr_ds = val_ds = te_ds = None
        torch.cuda.empty_cache()

    return results
