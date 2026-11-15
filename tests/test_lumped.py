"""Stage-1 lumped training: aggregation matrix, unit chain, checkpoint round trip."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import xtensor as xt

from diffhydro.pipelines.utils import mm_to_m3s

from data import init_dataset
from lumped import (LumpedModule, LumpedRunoffModel, build_lumped_inputs,
                    build_lumped_weight_matrix, lumped_dataset, runoff_state)
from model import InstantaneousRouter, load_runoff_state, check_runoff_init_config
from training import build_model, TEMP_RES_H
from conftest import ROUTING_FEATURES, upstream


def test_weight_matrix_rows_are_area_fractions(synth):
    g, nodes, area = synth["g"], synth["nodes"], synth["area"]
    W, areas = build_lumped_weight_matrix(g, pd.Index(nodes), area, synth["targets"])
    assert W.shape == (3, len(nodes))
    assert torch.allclose(W.sum(1), torch.ones(3))
    for r, d in enumerate(synth["targets"]):
        up = upstream(g, d)
        assert np.isclose(areas[r].item(), sum(area[n] for n in up))
        for j, n in enumerate(nodes):
            expected = area[n] / areas[r].item() if n in up else 0.0
            assert np.isclose(W[r, j].item(), expected)


class _Linear(nn.Module):
    """A memoryless linear 'runoff module': the case where lumped and
    distributed+instantaneous-sum are algebraically identical."""
    def forward(self, inp_dyn, inp_stat):
        v = 0.7 * inp_dyn.values.sum(-1) + 0.3 * inp_stat.values.sum(-1)[None, :, None] + 2.0
        return xt.DataTensor(v, dims=inp_dyn.dims[:-1],
                             coords={d: inp_dyn.coords[d] for d in inp_dyn.dims[:-1]})


def test_linear_lumped_equals_instantaneous_sum_of_catchments(synth):
    g, x, stat, area, targets = (synth["g"], synth["x"], synth["runoff_static"],
                                 synth["area"], synth["targets"])
    x_l, s_l, a_l = build_lumped_inputs(g, x, stat, area, targets)
    lumped = mm_to_m3s(_Linear()(x_l, s_l), a_l, TEMP_RES_H).values
    ds = init_dataset(g, x, synth["y"], stat, synth["routing_static"],
                      synth["channel_length"], area, targets, 30, 20,
                      irf_fn="instantaneous")
    per_cat = _Linear()(x.sel(spatial=ds.g.nodes), stat.sel(spatial=ds.g.nodes))
    summed = InstantaneousRouter()(mm_to_m3s(per_cat, ds.statics["cat_area"], TEMP_RES_H),
                                   ds.g).values
    assert torch.allclose(summed, lumped, rtol=1e-4, atol=1e-4)
    vol_gap = ((summed.sum(-1) - lumped.sum(-1)).abs() / lumped.sum(-1).abs()).max()
    assert vol_gap < 0.05


def test_lumped_module_trains_and_state_loads_into_rr_model(synth):
    g, x, stat, area, targets = (synth["g"], synth["x"], synth["runoff_static"],
                                 synth["area"], synth["targets"])
    x_l, s_l, a_l = build_lumped_inputs(g, x, stat, area, targets)
    tr = lumped_dataset(x_l, s_l, a_l, synth["y"], targets[:2], 30, 20)
    val = lumped_dataset(x_l, s_l, a_l, synth["y"], targets[2:], 30, 20)
    params = dict(hidden_size=8, num_layers=1)
    model = LumpedRunoffModel(3 + 2, params)
    module = LumpedModule(model, tr, val, tr, scheduler_type="cosine", n_epoch=2,
                          batch_size=2, inference_batch_size=2, device="cpu",
                          routing_lr=0.0, routing_wd=0.0, runoff_lr=1e-3, runoff_wd=0.0)
    assert [g_["name"] for g_ in module.opt.param_groups] == ["runoff"]
    tr_losses, _, _ = module.train(n_epoch=2, n_iter=2, track_val=False)
    assert len(tr_losses) == 2 and np.isfinite(np.concatenate(tr_losses)).all()
    ev = module.eval_variant("swa", device="cpu")
    assert len(ev["nse_val"]) == 1 and len(ev["nse_te"]) == 2
    state = runoff_state(module.weight_variants["swa"])
    rr, _ = build_model("instantaneous", ROUTING_FEATURES, 3, 2, params, "cpu")
    assert set(state) == set(rr.runoff_model.state_dict())
    rr.runoff_model.load_state_dict(state, strict=True)
    # the loaded distributed model reproduces the lumped LSTM per catchment
    xb = x_l.isel(time=slice(0, 10))
    out_lumped = model.runoff_model(xb, s_l).values
    out_rr = rr.runoff_model(xb, s_l).values
    assert torch.allclose(out_lumped, out_rr)


def test_load_runoff_state_from_pt_and_pkl(tmp_path, synth):
    params = dict(hidden_size=8, num_layers=1)
    rr, _ = build_model("hayami", ROUTING_FEATURES, 3, 2, params, "cpu")
    runoff_sd = {k: v.clone() for k, v in rr.runoff_model.state_dict().items()}
    pt = tmp_path / "00_lumped_fold3.pt"
    torch.save(dict(state_dict=runoff_sd, fold=3, seed=0,
                    config=dict(dynamic_keys=["a"], static_runoff_keys=["b"])), pt)
    state, meta = load_runoff_state(pt, fold=3)
    assert set(state) == set(runoff_sd) and meta["fold"] == 3
    with pytest.raises(ValueError):
        load_runoff_state(pt, fold=4)
    check_runoff_init_config(meta, ["a"], ["b"])
    with pytest.raises(ValueError):
        check_runoff_init_config(meta, ["a", "c"], ["b"])

    pkl = tmp_path / "00_noroute.pkl"
    full = rr.state_dict()
    pd.to_pickle(dict(config=dict(seed=0),
                      fold_results=[dict(fold=0, swa=dict(weights=full)),
                                    dict(fold=1, swa=dict())]), pkl)
    state, meta = load_runoff_state(pkl, fold=0)
    assert set(state) == set(runoff_sd)
    assert not any(k.startswith("routing_model") for k in state)
    rr.runoff_model.load_state_dict(state, strict=True)
    with pytest.raises(ValueError):
        load_runoff_state(pkl, fold=1)            # no weights saved
    with pytest.raises(ValueError):
        load_runoff_state(pkl, fold=7)            # no such fold
