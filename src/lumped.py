"""Lumped training of the runoff module -- stage 1 of the two-stage recipe.

The LSTM is trained per gauge on the area-weighted mean of the forcing and
the static descriptors over the gauge's whole upstream area, against the
gauge's discharge, with no routing at all.  The module is the *same*
``diffhydro.Runoff`` as ``RRModel.runoff_model`` (same inputs, same
normalisation, same softplus output), and the unit chain is ``RRModel.
forward``'s -- mm per time step, converted to m3/s through the *total
upstream area* with ``mm_to_m3s`` and compared with the y-normalised target --
so the state dict drops into ``RRModel.runoff_model`` unchanged, and applying
it per catchment then summing upstream instantaneously reproduces the lumped
volume exactly for a linear model (``tests/test_lumped.py``).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xtensor as xt

import diffhydro as dh
import diffhydro.pipelines as dhp
from diffhydro.pipelines.utils import mm_to_m3s

from data import TimeMaskedDataset
from model import RUNOFF_PREFIX
from training import RRModule, TEMP_RES_H


def _spatial_index(dt: xt.DataTensor) -> pd.Index:
    sp = dt.coords["spatial"]
    return pd.Index(sp.detach().cpu().numpy() if isinstance(sp, torch.Tensor) else sp)


def build_lumped_weight_matrix(g, all_spatial: pd.Index, area: pd.Series,
                               gauge_nodes):
    """Area-weighted aggregation matrix ``W`` (n_gauges, n_nodes) over each
    gauge's upstream set (ancestors and the gauge's own catchment), and the
    total upstream area (km2) per gauge.  Rows sum to one."""
    W = torch.zeros(len(gauge_nodes), len(all_spatial), dtype=torch.float32)
    col = pd.Series(np.arange(len(all_spatial)), index=all_spatial)
    areas = torch.zeros(len(gauge_nodes), dtype=torch.float32)
    for row, node in enumerate(gauge_nodes):
        upstream = [n for n in (nx.ancestors(g, node) | {node}) if n in col.index]
        w = torch.tensor([float(area[n]) for n in upstream], dtype=torch.float32)
        W[row, torch.as_tensor(col.loc[upstream].to_numpy())] = w / w.sum()
        areas[row] = w.sum()
    return W, areas


def build_lumped_inputs(g, x: xt.DataTensor, runoff_statics: xt.DataTensor,
                        area: pd.Series, gauge_nodes, chunk_size: int = 500):
    """``(x_l, s_l, area_l)``: (1, n_gauges, T, V) lumped dynamics, (n_gauges,
    S) lumped statics, (n_gauges,) total upstream areas.

    The mean is taken over the *normalised* inputs the distributed model sees;
    normalisation being affine per variable and the weights summing to one,
    this equals normalising the area-weighted mean of the raw fields.
    """
    all_spatial = _spatial_index(x)
    W, area_l = build_lumped_weight_matrix(g, all_spatial, area, gauge_nodes)
    _, n_nodes, T, V = x.shape
    x_mat = x.values[0].reshape(n_nodes, T * V)
    stat = runoff_statics.sel(spatial=all_spatial).values.float()

    x_chunks, s_chunks = [], []
    for s0 in range(0, len(gauge_nodes), chunk_size):
        Wc = W[s0:s0 + chunk_size]
        x_chunks.append(Wc @ x_mat)
        s_chunks.append(Wc @ stat)
    gauge_idx = pd.Index(gauge_nodes)
    x_l = xt.DataTensor(
        torch.cat(x_chunks).reshape(1, len(gauge_nodes), T, V),
        dims=("batch", "spatial", "time", "variable"),
        coords={"batch": x.coords["batch"], "spatial": gauge_idx,
                "time": x.coords["time"], "variable": x.coords["variable"]})
    s_l = xt.DataTensor(
        torch.cat(s_chunks), dims=("spatial", "variable"),
        coords={"spatial": gauge_idx,
                "variable": runoff_statics.coords["variable"]})
    return x_l, s_l, area_l


class LumpedRunoffModel(nn.Module):
    """The runoff module alone, on lumped inputs, with ``RRModel.forward``'s
    unit chain: ``mm_to_m3s(runoff_mm, total_upstream_area, temp_res_h)``.

    Same call signature as ``RRModel.forward`` (``g`` and
    ``additional_params`` are ignored) so ``RRModule`` drives it unchanged.
    """

    def __init__(self, input_size, runoff_params, temp_res_h=TEMP_RES_H):
        super().__init__()
        self.temp_res_h = temp_res_h
        self.runoff_model = dh.Runoff(input_size=input_size, softplus=True,
                                      **runoff_params)

    def forward(self, inp_dyn, inp_stat, g=None, cat_area=None,
                additional_params=None):
        runoff_mm = self.runoff_model(inp_dyn, inp_stat)
        return mm_to_m3s(runoff_mm, cat_area, self.temp_res_h)


class LumpedModule(RRModule):
    """``RRModule`` without a graph: the statics carry the per-gauge total
    upstream area (``cat_area``) and the lumped descriptors (``x_stat``)."""

    def run_model(self, ds, x):
        x_stat = ds.statics["x_stat"].to(x.device)
        cat_area = ds.statics["cat_area"].to(x.device)
        return self.model(x, x_stat, None, cat_area)


def lumped_dataset(x_l, s_l, area_l, y, nodes, init_window, pred_len,
                   train_days=None):
    """A ``BaseDataset`` of lumped inputs for ``nodes`` (a ``TimeMaskedDataset``
    when ``train_days`` is given, i.e. for the training set of a block run)."""
    rows = torch.as_tensor(pd.Index(list(_spatial_index(x_l))).get_indexer(nodes))
    if (rows < 0).any():
        raise KeyError("nodes missing from the lumped inputs")
    statics = {"cat_area": area_l[rows], "x_stat": s_l.sel(spatial=nodes),
               "channel_dist": torch.zeros(len(nodes))}
    kwargs = dict(x=x_l.sel(spatial=nodes), y=y.sel(spatial=nodes), g=None,
                  init_len=init_window, pred_len=pred_len, statics=statics)
    if train_days is None:
        return dhp.BaseDataset(**kwargs)
    return TimeMaskedDataset(train_days=train_days, **kwargs)


def runoff_state(state_dict: dict) -> dict:
    """The ``runoff_model.*`` entries of a ``LumpedRunoffModel`` (or
    ``RRModel``) state dict, with the prefix stripped -- the layout
    ``RRModel.runoff_model.load_state_dict`` expects."""
    return {k[len(RUNOFF_PREFIX):]: v for k, v in state_dict.items()
            if k.startswith(RUNOFF_PREFIX)}
