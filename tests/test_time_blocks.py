"""Temporal holdout: block masks, the masked training dataset, the untouched
evaluation dataset."""
import numpy as np
import pandas as pd
import pytest
import torch

import diffhydro.pipelines as dhp
from diffhydro.pipelines.base import init_optim_dl

from data import (TimeMaskedDataset, block_mask, init_split_dataset,
                  mask_targets, time_index)
from training import RRModule, build_model
from conftest import ROUTING_FEATURES

INIT, PRED = 30, 20


def test_block_mask_is_decided_on_label_dates(synth):
    t = synth["times"]
    m = block_mask(t, synth["block"])
    assert int(m.sum()) == 30 + 31 + 31 + 30
    assert m[t.get_loc(pd.Timestamp("2008-06-01 12:00"))]
    assert m[t.get_loc(pd.Timestamp("2008-09-30 12:00"))]      # inclusive end, 12Z label
    assert not m[t.get_loc(pd.Timestamp("2008-05-31 12:00"))]
    assert not m[t.get_loc(pd.Timestamp("2008-10-01 12:00"))]
    assert torch.equal(block_mask(time_index(synth["y"]), synth["block"]), m)


def _split(synth, holdout, irf="instantaneous"):
    y = synth["y"]
    return init_split_dataset(
        synth["g"], synth["x"], y, synth["runoff_static"], synth["routing_static"],
        synth["channel_length"], synth["area"],
        synth["targets"], synth["targets"], synth["targets"], INIT, PRED,
        irf_fn=irf, y_train=mask_targets(y, holdout), train_days=~holdout)


def test_heldout_targets_nan_in_train_present_in_test(synth):
    holdout = block_mask(synth["times"], synth["block"])
    tr, val, te = _split(synth, holdout)
    assert isinstance(tr, TimeMaskedDataset)
    assert type(val) is dhp.BaseDataset and type(te) is dhp.BaseDataset
    assert torch.isnan(tr.y.values[..., holdout]).all()
    assert torch.isfinite(te.y.values[..., holdout]).all()
    assert torch.isfinite(val.y.values[..., holdout]).all()
    # outside the block the training targets are the originals
    assert torch.equal(torch.nan_to_num(tr.y.values[..., ~holdout]),
                       torch.nan_to_num(te.y.values[..., ~holdout]))
    # y_var is computed on training days only
    y_tr_days = te.y.values[..., ~holdout]
    expected = torch.from_numpy(np.nanvar(y_tr_days.numpy(), axis=-1)).float()
    assert torch.allclose(tr.y_var.values.squeeze(), expected.squeeze(), atol=1e-5)
    assert not torch.allclose(tr.y_var.values, te.y_var.values)


def test_sampled_windows_stay_inside_training_blocks(synth):
    holdout = block_mask(synth["times"], synth["block"])
    tr, _, _ = _split(synth, holdout)
    n_time = len(synth["times"])
    brute = [s for s in range(n_time - (INIT + PRED))
             if not holdout[s + INIT:s + INIT + PRED].any()]
    assert tr.valid_starts.tolist() == brute
    assert len(tr) == len(brute) < n_time - (INIT + PRED)
    for i in range(len(tr)):
        _, yb = tr[i]
        pred = yb.values[..., INIT:]
        assert pred.shape[-1] == PRED
        assert torch.isfinite(pred).any(), "an all-NaN target window was sampled"
        assert torch.isfinite(pred[0, :2]).all()      # complete records: no NaN at all
    # composes with the stock training DataLoader (random indices over len(ds))
    xb, yb = next(iter(init_optim_dl(tr, 2)))
    assert xb.shape[0] == 2 and yb.shape[-1] == INIT + PRED


def test_no_window_when_training_blocks_too_short(synth):
    days = torch.zeros(len(synth["times"]), dtype=torch.bool)
    days[:INIT + 5] = True                         # only 5 predictable days
    with pytest.raises(ValueError):
        TimeMaskedDataset(synth["x"], synth["y"], INIT, PRED, train_days=days)


def test_training_step_and_full_series_evaluation_cpu(synth):
    holdout = block_mask(synth["times"], synth["block"])
    tr, val, te = _split(synth, holdout)
    model, _ = build_model("instantaneous", ROUTING_FEATURES, 3, 2,
                           dict(hidden_size=8, num_layers=1), "cpu")
    module = RRModule(model, tr, val, te, scheduler_type="cosine", n_epoch=1,
                      batch_size=2, inference_batch_size=2, device="cpu",
                      routing_lr=1e-4, routing_wd=0.0, runoff_lr=1e-3, runoff_wd=0.0)
    losses = module.train_epoch(n_iter=2, device="cpu")
    assert np.isfinite(losses).all()
    yte, ote = module.extract_test(device="cpu", batch_size=2)
    # the evaluation series is continuous and covers the held-out block
    tidx = pd.DatetimeIndex(yte.coords["time"])
    assert tidx[0] == synth["times"][INIT]
    assert tidx[-1] == synth["times"][-1]              # the tail is not dropped
    assert len(tidx) == len(synth["times"]) - INIT
    assert torch.equal(torch.nan_to_num(yte.values), torch.nan_to_num(te.y.values[0, :, INIT:]))
    assert block_mask(tidx, synth["block"]).sum() == holdout.sum()
    assert torch.isfinite(ote.values).all()


def test_gauge_confined_to_heldout_block_is_dropped_and_loss_finite(synth, capsys):
    """A training gauge with data only inside the held-out block has a NaN
    variance on training days: it must leave the training set and the loss
    must stay finite."""
    from training import run_experiments, clipped_nse_loss
    import xtensor as xt
    y = synth["y"]
    holdout = block_mask(synth["times"], synth["block"])
    yv = y.values.clone()
    yv[0, 2, ~holdout] = float("nan")            # third gauge: block-only record
    y2 = xt.DataTensor(yv, dims=y.dims, coords=y.coords)
    y_train = mask_targets(y2, holdout)
    assert torch.isnan(y_train.values[0, 2]).all()
    # the loss itself is finite with a NaN-variance station
    o = xt.DataTensor(torch.rand(1, 3, 10), dims=y.dims, coords={"batch": xt.arange_index(1),
                      "spatial": y.coords["spatial"], "time": xt.arange_index(10)})
    yy = xt.DataTensor(torch.rand(1, 3, 10), dims=o.dims, coords=o.coords)
    var = xt.DataTensor(torch.tensor([[1.0, 2.0, float("nan")]]), dims=("batch", "spatial"),
                        coords={"batch": xt.arange_index(1), "spatial": y.coords["spatial"]})
    assert torch.isfinite(clipped_nse_loss(o, yy, var, station_loss_clip=5.0))
    t = synth["targets"]
    res = run_experiments(
        synth["g"], synth["x"], y2, synth["runoff_static"], synth["routing_static"],
        synth["channel_length"], synth["area"], [(t, t, t)], ROUTING_FEATURES,
        device="cpu", batch_size=1, inference_batch_size=2, n_epoch=1, n_iter=2,
        init_window=INIT, pred_len=PRED, runoff_params=dict(hidden_size=8, num_layers=1),
        runoff_lr=1e-3, runoff_wd=0.0, routing_lr=1e-4, routing_wd=0.0,
        scheduler_type="cosine", scheduler_step_size=None, scheduler_gamma=0.1,
        clip_grad_norm=1.0, station_loss_clip=5.0, variants=("swa",),
        irf_fn="instantaneous", save_series=True,
        holdout_mask=holdout, holdout_bounds=synth["block"])
    out = capsys.readouterr().out
    assert "1 training gauge(s) without any target inside the training blocks" in out
    fr = res[0]
    assert np.isfinite(np.concatenate(fr["tr_losses"])).all()
    assert len(fr["swa"]["nse_te"]) == 3                 # still evaluated
    assert fr["swa"]["nse_te_holdout"].notna().all()
