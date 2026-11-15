"""Continuous forward passes of a trained model over the whole period.

``continuous_series`` runs a model over a dataset with the same window
striding as the training code's evaluation (``training.RRModule.
_extract_full_ts``: windows of ``init_window`` warm-up days + ``pred_len``
scored days, strided by ``pred_len``, plus one final window ending on the last
day), so it reproduces the series a training run reports.  ``simulate`` runs
it basin by basin over any set of target nodes -- routing is basin-local, so
splitting the network into groups of whole basins bounds GPU memory without
changing a single value.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import torch
import xtensor as xt

import diffhydro.pipelines as dhp
from diffhydro.pipelines.base import init_inference_dl
from diffhydro.pipelines.utils import collate_fn

from data import init_dataset, time_index


def _flatten(batches, ds, spatial):
    """(spatial, time) from per-batch (batch, spatial, time) window outputs, as
    ``diffhydro.pipelines.base.format_batched_series`` does."""
    data = torch.cat([b.transpose("spatial", "batch", "time").values.contiguous()
                      .view(b.sizes["spatial"], -1) for b in batches], -1)
    return xt.DataTensor(data, dims=("spatial", "time"),
                         coords={"spatial": spatial,
                                 "time": ds.y.coords["time"][ds.init_len:data.shape[-1] + ds.init_len]})


def continuous_series(model, ds, device, batch_size=8, with_runoff=False):
    """``(discharge, runoff)`` DataTensors (spatial = the dataset's targets,
    time = every day after the warm-up) in the model's normalised units;
    ``runoff`` (the catchment's own runoff before routing, captured from the
    runoff module's output) is ``None`` unless ``with_runoff``."""
    model.eval()
    # the arguments of diffhydro's RRModule.run_model, which training evaluates with
    g = ds.g.to(device)
    x_stat = ds.statics["x_stat"].to(device)
    cat_area = ds.statics["cat_area"].to(device)
    channel_dist = ds.statics["channel_dist"].to(device)
    targets = ds.y.coords["spatial"]
    captured = {}
    handle = model.runoff_model.register_forward_hook(
        lambda m, args, out: captured.__setitem__("r", out)) if with_runoff else None

    def run(x):
        o = model(x.to(device), x_stat, g, cat_area, channel_dist).sel(spatial=targets)
        r = captured.pop("r").sel(spatial=targets) if with_runoff else None
        return (o.isel(time=slice(ds.init_len, None)),
                r.isel(time=slice(ds.init_len, None)) if with_runoff else None)

    outs, runs = [], []
    try:
        with torch.no_grad():
            for x, _ in init_inference_dl(ds, batch_size):
                o, r = run(x)
                outs.append(o)
                runs.append(r)
            q = _flatten(outs, ds, targets)
            ro = _flatten(runs, ds, targets) if with_runoff else None
            n_time = len(ds.y.coords["time"])
            missing = n_time - (ds.init_len + q.shape[-1])
            if missing > 0:
                xw, _ = collate_fn([dhp.BaseDataset.__getitem__(ds, n_time - ds.total_len)])
                ow, rw = run(xw)
                tail_t = ds.y.coords["time"][n_time - missing:]

                def tail(w):
                    return xt.DataTensor(w.isel(time=slice(-missing, None)).values[0],
                                         dims=("spatial", "time"),
                                         coords={"spatial": targets, "time": tail_t})
                q = xt.concat([q, tail(ow)], "time")
                if with_runoff:
                    ro = xt.concat([ro, tail(rw)], "time")
    finally:
        if handle is not None:
            handle.remove()
    return q, ro


def basin_groups(g, n_groups, nodes=None):
    """Split the basins (weakly connected trees) holding ``nodes`` (default:
    every node) into ``n_groups`` groups of similar size; returns, per group,
    the requested nodes it holds."""
    nodes = list(g.nodes) if nodes is None else list(nodes)
    wanted = set(nodes)
    roots = [n for n in g.nodes if g.out_degree(n) == 0]
    basins = {r: ({r} | nx.ancestors(g, r)) & wanted for r in roots}
    basins = {r: b for r, b in basins.items() if b}
    order = sorted(basins, key=lambda r: -len(nx.ancestors(g, r)))
    groups, sizes = [[] for _ in range(n_groups)], [0] * n_groups
    for r in order:
        k = int(np.argmin(sizes))
        groups[k].append(r)
        sizes[k] += len(nx.ancestors(g, r)) + 1
    pos = {n: i for i, n in enumerate(nodes)}
    return [sorted((n for r in grp for n in basins[r]), key=pos.get) for grp in groups if grp]


def simulate(model, spec, data, targets, device, n_groups=1, batch_size=8,
             with_runoff=False, time_slice=None, log=print, y_std=None):
    """Simulated discharge (m3/s) and, optionally, runoff (mm/day) at ``targets``.

    ``data``: the tuple returned by ``data.data_loading``; ``spec``: the model
    file's spec (``weights.read_model_file(path)["spec"]``); ``time_slice``:
    optional ``(first, last)`` day labels of the output, the warm-up being
    taken before ``first``; ``y_std``: the discharge scale the model was
    trained with (its model file's ``norm["y_std"]``), which converts its
    outputs to m3/s -- by default the scale recomputed from the dataset.  Returns ``(times, discharge, runoff)`` with
    ``discharge``/``runoff`` as ``(time, target)`` float32 arrays in the order
    of ``targets`` (``runoff`` is ``None`` unless ``with_runoff``).
    """
    (g, x, y, _xm, _xs, _y_std, runoff_statics, routing_statics,
     channel_length, area, _splits) = data
    ystd = float(y_std) if y_std is not None else float(_y_std.values)
    targets = [int(t) for t in targets]
    times = pd.DatetimeIndex(time_index(x))
    # observations where they exist, NaN elsewhere: the target tensor only
    # fixes the output's spatial axis and length
    y_all = xt.DataTensor(torch.full((1, len(targets), len(times)), float("nan")),
                          dims=("batch", "spatial", "time"),
                          coords={"batch": y.coords["batch"], "spatial": pd.Index(targets),
                                  "time": x.coords["time"]})
    init_window = int(spec["init_window"])
    if time_slice is not None:
        keep = (times >= pd.Timestamp(time_slice[0])) & (times <= pd.Timestamp(time_slice[1]))
        first, last = int(np.flatnonzero(keep)[0]), int(np.flatnonzero(keep)[-1])
        if first < init_window:
            raise ValueError(f"{time_slice[0]} leaves fewer than {init_window} warm-up days")
        sel = slice(first - init_window, last + 1)
        x, y_all = x.isel(time=sel), y_all.isel(time=sel)
    out_times = pd.DatetimeIndex(time_index(x))[init_window:]
    pos = {n: i for i, n in enumerate(targets)}
    Q = np.full((len(out_times), len(targets)), np.nan, np.float32)
    R = np.full((len(out_times), len(targets)), np.nan, np.float32) if with_runoff else None
    groups = basin_groups(g, n_groups, targets)
    for k, grp in enumerate(groups):
        ds = init_dataset(g, x, y_all.sel(spatial=grp), runoff_statics, routing_statics,
                          channel_length, area, grp, init_window, int(spec["pred_len"]),
                          route_src_reach=bool(spec["route_src_reach"]), irf_fn=spec["irf_fn"])
        q, ro = continuous_series(model, ds, device, batch_size=batch_size,
                                  with_runoff=with_runoff)
        assert pd.DatetimeIndex(time_index(q)).equals(out_times)
        cols = [pos[int(n)] for n in q.coords["spatial"]]
        Q[:, cols] = q.values.detach().cpu().numpy().T * ystd
        if with_runoff:
            R[:, cols] = ro.values.detach().cpu().numpy().T * ystd
        log(f"  group {k + 1}/{len(groups)}: {len(grp)} targets")
        del ds, q, ro
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out_times, Q, R
