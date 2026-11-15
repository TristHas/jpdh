"""The coupled model: LSTM runoff module, routing-parameter MLP and the routing kernels."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xtensor as xt

import diffhydro.pipelines as dhp
from diffhydro.pipelines.routing import LearnedRouter
from diffhydro.pipelines.utils import PARAMS_BOUNDS, mm_to_m3s


#: the no-routing scheme of the ablation: every catchment's outlet runoff
#: reaches each downstream node in the same time step.  Not a DiffRoute IRF --
#: see InstantaneousRouter.
INSTANTANEOUS = "instantaneous"
IRF_CHOICES = ["linear_storage", "nash_cascade", "muskingum", "pure_lag", "hayami",
               INSTANTANEOUS]


def riv_tree_irf(irf_fn):
    """The IRF name a ``RivTree`` is built with for a given ``--irf_fn``.

    DiffRoute sizes ``g.params`` from a registered IRF, so the instantaneous
    scheme borrows ``pure_lag``'s one-column layout; the router never reads
    those parameters.
    """
    return "pure_lag" if irf_fn == INSTANTANEOUS else irf_fn


# ---------------------------------------------------------------------------
# No-routing arm: exact instantaneous upstream sum
# ---------------------------------------------------------------------------

def upstream_sum_indices(g):
    """``(rows, cols)`` of the 0/1 aggregation matrix ``A`` (n_out x n_nodes)
    of a ``RivTree``: ``A[r, c] = 1`` iff node ``c`` is the r-th output node
    or one of its ancestors.  Positions follow ``g.nodes_idx``, rows follow
    ``g.output_reach`` (every node in graph order when it is ``None``).
    """
    nodes_idx = g.nodes_idx
    outputs = list(g.output_reach) if g.output_reach is not None \
        else list(nodes_idx.index)
    rows, cols = [], []
    for r, d in enumerate(outputs):
        up = list(nx.ancestors(g.g, d) | {d})
        cols.append(nodes_idx.loc[up].to_numpy(dtype=np.int64))
        rows.append(np.full(len(up), r, dtype=np.int64))
    return (torch.from_numpy(np.concatenate(rows)),
            torch.from_numpy(np.concatenate(cols)))


def attach_upstream_sum(g):
    """Register the upstream-sum index pair on a ``RivTree`` (as buffers, so
    ``g.to(device)`` carries them along with the rest of the graph)."""
    rows, cols = upstream_sum_indices(g)
    g.register_buffer("upstream_rows", rows)
    g.register_buffer("upstream_cols", cols)
    return g


class InstantaneousRouter(nn.Module):
    """Routing operator of the no-routing arm: ``y[d] = sum_{s upstream of or
    equal to d} x[s]`` at every time step.

    This is what DiffRoute's ``route_src_reach=False`` semantics reduce to when
    every reach IRF is a Dirac: a catchment's own runoff is already at its
    outlet (the identity diagonal) and every strictly downstream path passes it
    on undelayed and unattenuated.  It is evaluated as a sparse 0/1 matrix
    product on the tree's ancestor relation rather than through DiffRoute's
    ``pure_lag`` kernel at zero delay, because that kernel goes through an FFT
    path closure, a relu and an hourly-to-daily triangular resampling and is
    *not* the identity (``tests/test_instantaneous.py``), whereas the sum is
    exact, deterministic, CPU-capable and needs no kernel.  No parameters:
    ``--route_src_reach`` is inert under this scheme.

    Same call contract as ``diffhydro.LTIRouter``: a ``(batch, spatial, time)``
    DataTensor in, a DataTensor whose spatial axis is ``g.output_reach`` out.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    def _matrix(self, g, device, dtype):
        key = (str(device), dtype)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is g:
            return cached[1]
        if not hasattr(g, "upstream_rows"):
            attach_upstream_sum(g)
        rows, cols = g.upstream_rows.to(device), g.upstream_cols.to(device)
        A = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            torch.ones(len(rows), device=device, dtype=dtype),
            (g.n_out, len(g.nodes_idx))).coalesce()
        self._cache[key] = (g, A)
        return A

    def forward(self, runoff, g, params=None, cluster_idx=None):
        x = runoff.values
        *lead, n_nodes, n_time = x.shape
        A = self._matrix(g, x.device, x.dtype)
        flat = x.reshape(-1, n_nodes, n_time).permute(1, 0, 2).reshape(n_nodes, -1)
        y = torch.sparse.mm(A, flat)                       # (n_out, B*T)
        y = y.reshape(g.n_out, -1, n_time).permute(1, 0, 2).reshape(*lead, g.n_out, n_time)
        coords = runoff.coords if g.output_reach is None \
            else {**runoff.coords, "spatial": list(g.output_reach)}
        return xt.DataTensor(y, dims=runoff.dims, coords=coords, name="discharge")


# ---------------------------------------------------------------------------
# Coupled model
# ---------------------------------------------------------------------------

class JPDHRRModel(dhp.RRModel):
    """``dhp.RRModel`` with the two switches the ablations need.

    * ``irf_name="instantaneous"`` swaps the routing operator for
      ``InstantaneousRouter`` (no-routing arm).
    * ``freeze_runoff()`` fixes the runoff module (two-stage recipe): its
      parameters stop requiring gradients, so ``training.RRModule`` leaves them
      out of the optimizer, and its forward runs under ``torch.no_grad()``:
      only the autograd graph through the LSTM -- the dominant memory and
      time cost of a step -- is skipped.  The function is unchanged; the
      values match the trainable path to float32 kernel round-off (~1e-7:
      cuDNN and MKL may select an inference kernel when no gradient is
      needed, see ``tests/test_two_stage.py``).

    ``dhp.RRModel`` builds the runoff module
    (``Runoff(input_size=n_dyn + n_stat, softplus=True, ...)``); the unit
    chain of ``forward`` (mm per step -> m3/s through the catchment area ->
    routing) is what ``lumped.LumpedRunoffModel`` mirrors.
    """

    def __init__(self, param_model, input_size, max_delay, dt,
                 irf_name, temp_res_h, runoff_params={}, routing_params={}):
        super().__init__(
            param_model, input_size=input_size, max_delay=max_delay, dt=dt,
            irf_name=riv_tree_irf(irf_name), temp_res_h=temp_res_h,
            runoff_params=runoff_params, routing_params=routing_params,
        )
        self.irf_name = irf_name
        if irf_name == INSTANTANEOUS:
            self.routing_model = InstantaneousRouter()
        self.runoff_frozen = False

    def freeze_runoff(self):
        for p in self.runoff_model.parameters():
            p.requires_grad_(False)
        self.runoff_frozen = True
        return self

    def forward(self, inp_dyn, inp_stat, g, cat_area, additional_params=None):
        if self.runoff_frozen:
            with torch.no_grad():
                runoff_mm = self.runoff_model(inp_dyn, inp_stat)
        else:
            runoff_mm = self.runoff_model(inp_dyn, inp_stat)
        runoff_m3s = mm_to_m3s(runoff_mm, cat_area, self.temp_res_h)
        return self.routing_model(runoff_m3s, g, additional_params)


#: DiffHydro's initial-value words, as the logit added to the parameter
#: module's output before the sigmoid (``format_param_bounds``)
INIT_LOGIT = {"low": -3.0, "mid": 0.0, "high": 3.0}


def _slot_scale(irf_name, slot, temp_res_h):
    """Factor from the hourly units of ``PARAMS_BOUNDS`` to the model's units
    (``diffhydro.pipelines.utils.infer_param_bounds``' rule)."""
    if irf_name == "hayami":
        return float(temp_res_h)
    if irf_name in ("linear_storage", "nash_cascade") or (irf_name == "muskingum" and slot == "k"):
        return 1.0 / float(temp_res_h)
    return 1.0


def override_router_buffers(router, irf_name, temp_res_h, bounds=None, init=None):
    """Replace a ``LearnedRouter``'s bound and initial-value buffers.

    ``bounds``: ``{slot: (min, max)}`` in the hourly units of
    ``PARAMS_BOUNDS`` (linear-storage tau in hours, pure-lag delay in steps,
    ...); ``init``: one of ``INIT_LOGIT`` applied to every slot.  The buffers
    are what the state dict saves, so a run's fields are read back with the
    bounds it trained under.
    """
    if not bounds and init is None:
        return router
    if not all(hasattr(router, b) for b in ("offset", "range", "param_init")):
        raise ValueError(f"{type(router).__name__} does not read its parameters through "
                         f"sigmoid-bounded buffers")
    slots = list(PARAMS_BOUNDS[irf_name].columns)
    offset, rng, init_logit = router.offset.clone(), router.range.clone(), router.param_init.clone()
    for slot, (lo, hi) in (bounds or {}).items():
        if slot not in slots:
            raise ValueError(f"{irf_name} has no parameter {slot!r}; slots: {slots}")
        if not (0 <= lo < hi):
            raise ValueError(f"{slot}: bounds must satisfy 0 <= min < max, got {lo}, {hi}")
        j = slots.index(slot)
        k = _slot_scale(irf_name, slot, temp_res_h)
        offset[j], rng[j] = lo * k, (hi - lo) * k
    if init is not None:
        init_logit[:] = INIT_LOGIT[init]
    router.offset.copy_(offset)
    router.range.copy_(rng)
    router.param_init.copy_(init_logit)
    return router


def parse_routing_bounds(spec):
    """``"tau=0,9.9;k=0.1,6"`` -> ``{"tau": (0.0, 9.9), "k": (0.1, 6.0)}``."""
    out = {}
    for item in (spec or "").split(";"):
        item = item.strip()
        if not item:
            continue
        name, vals = item.split("=")
        lo, hi = (float(v) for v in vals.split(","))
        out[name.strip()] = (lo, hi)
    return out


def build_param_model(irf_fn, features):
    """The routing-parameter module of a scheme: an MLP from the routing
    descriptors to the kernel's parameters, or ``None`` for the parameter-less
    instantaneous scheme."""
    if irf_fn == INSTANTANEOUS:
        return None
    return dhp.MLP(len(features), len(PARAMS_BOUNDS[irf_fn].columns))


# ---------------------------------------------------------------------------
# Two-stage recipe: runoff-module checkpoints
# ---------------------------------------------------------------------------

RUNOFF_PREFIX = "runoff_model."


def resolve_runoff_init(template: str, seed: int, fold: int) -> Path:
    """``--runoff_init`` templates may use ``{seed}`` / ``{seed:02d}`` and
    ``{fold}``, e.g. ``results/lumped/{seed:02d}_lumped_fold{fold}.pt``."""
    return Path(template.format(seed=seed, fold=fold))


def load_runoff_state(path, fold=None, variant="swa"):
    """State dict of a runoff module, keys relative to ``runoff_model``.

    Accepts a lumped checkpoint written by ``experiments/train_lumped.py``
    (``.pt`` with ``state_dict``; one file per fold, so ``fold`` is only
    checked against what it records) or a ``train.py`` result pickle
    (``.pkl`` run with ``--save_weights``: the weights of ``variant`` for
    ``fold``, filtered to the ``runoff_model.`` entries and stripped of the
    prefix).  Returns ``(state_dict, meta)``.
    """
    path = Path(path)
    if path.suffix == ".pt":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["state_dict"]
        if fold is not None and ck.get("fold") is not None and ck["fold"] != fold:
            raise ValueError(f"{path} was trained on fold {ck['fold']}, not {fold}")
        meta = {k: ck.get(k) for k in ("fold", "seed", "config", "nse_train_median",
                                       "nse_val_median")}
    elif path.suffix == ".pkl":
        run = pd.read_pickle(path)
        matches = [fr for fr in run["fold_results"] if fr["fold"] == fold]
        if not matches:
            raise ValueError(f"{path} has no fold {fold}")
        weights = matches[0][variant].get("weights")
        if weights is None:
            raise ValueError(f"{path}: fold {fold} '{variant}' carries no weights "
                             f"(run train.py with --save_weights)")
        state = {k[len(RUNOFF_PREFIX):]: v for k, v in weights.items()
                 if k.startswith(RUNOFF_PREFIX)}
        meta = {"fold": fold, "seed": run["config"].get("seed"),
                "config": run["config"], "variant": variant}
    else:
        raise ValueError(f"--runoff_init: unknown checkpoint type {path.suffix}")
    if not state:
        raise ValueError(f"{path}: empty runoff state")
    return state, meta


def check_runoff_init_config(meta: dict, dynamic_keys, static_runoff_keys):
    """A stage-1 checkpoint must have seen the same inputs as the stage-2 run
    (same variables in the same order): a mismatch would silently feed the
    LSTM's weights with the wrong columns."""
    cfg = meta.get("config") or {}
    for key, ours in (("dynamic_keys", list(dynamic_keys)),
                      ("static_runoff_keys", list(static_runoff_keys))):
        theirs = cfg.get(key)
        if theirs is not None and list(theirs) != ours:
            raise ValueError(f"--runoff_init checkpoint used {key}={list(theirs)}, "
                             f"this run uses {ours}")


# ---------------------------------------------------------------------------
# Output-bias initialisation (cold start of the clipped NSE loss)
# ---------------------------------------------------------------------------

def mean_specific_runoff(y, g, area, gauges, temp_res_h=24):
    """Median over ``gauges`` of mean(target) / (upstream area x m3/s-per-mm
    factor): the runoff, in the target's (y-normalised) units per time step,
    that reproduces each gauge's mean discharge -- what the runoff module
    should emit at initialisation to start inside the loss clip."""
    from diffhydro.pipelines.utils import M3S_TO_MMKM2
    vals = []
    yv = y.sel(spatial=list(gauges)).values
    for i, n in enumerate(gauges):
        up = nx.ancestors(g, n) | {n}
        a_tot = float(sum(area[m] for m in up))
        mean_q = torch.nanmean(yv[..., i, :]).item()
        if np.isfinite(mean_q) and a_tot > 0:
            vals.append(mean_q / (a_tot * M3S_TO_MMKM2 / temp_res_h))
    if not vals:
        raise ValueError("no gauge with a finite mean target")
    return float(np.median(vals))


def init_output_bias(runoff_model, target):
    """Set the runoff module's output-layer bias so that its initial output is
    ``target`` (softplus inverse); the output weights are zeroed so the start
    is exactly that constant.  A fresh ``LSTMModel`` emits softplus(~0) ~ 0.69
    mm per step, far above the typical mean specific runoff of Japanese basins
    in the normalised units the loss uses, so every station would start above
    the per-station loss clip (5), where the gradient is exactly zero, and
    training would stall for tens of epochs.  ``train.py --init_output_bias
    mean_runoff`` (the default)."""
    linear = runoff_model.core.linear
    target = max(float(target), 1e-6)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(float(np.log(np.expm1(target))))
    return target
