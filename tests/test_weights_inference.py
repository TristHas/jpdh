"""Model files (src/weights.py) and continuous forward passes (src/inference.py)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data import init_dataset, init_split_dataset
from inference import basin_groups, continuous_series, simulate
from training import RRModule, build_model
from weights import check_norm, export_run, load_model, read_model_file
from conftest import ROUTING_FEATURES

PARAMS = dict(hidden_size=8, num_layers=1)
INIT, PRED = 30, 20


def _fake_run(tmp_path, n_folds=2):
    """A train.py-like pickle with saved weights, for the default keys."""
    model, _ = build_model("linear_storage", ["channel_length_km"], 12, 25, PARAMS, "cpu",
                           routing_bounds={"tau": (0.01, 9.9)}, routing_init="low")
    cfg = dict(irf_fn="linear_storage", routing_features=["channel_length_km"],
               routing_bounds="tau=0.01,9.9", routing_init="low",
               dynamic_keys=["gauge_prcp_4h_bin", "msm_a_temp_4h_bin"],
               static_runoff_keys=["topo", "land_use", "season_msm"],
               runoff_hidden_size=8, runoff_num_layers=1, init_window=365, pred_len=100,
               route_src_reach=False, seed=0, exp_name="demo", y_std=12.5,
               libs={"diffhydro": "x"})
    x_mean = pd.Series(np.arange(12, dtype=float), index=[f"v{i}" for i in range(12)])
    folds = [dict(fold=k, swa=dict(nse_te=pd.Series([0.5, 0.6], index=[10 + k, 20 + k]),
                                    nse_val=pd.Series([0.7], index=[30 + k]),
                                    weights=model.state_dict()))
             for k in range(n_folds)]
    path = tmp_path / "00_demo.pkl"
    pd.to_pickle(dict(config=cfg, fold_results=folds, x_mean=x_mean, x_std=x_mean + 1,
                      y_std=12.5), path)
    return path, model


def test_export_and_reload_round_trip(tmp_path):
    pkl, model = _fake_run(tmp_path)
    paths = export_run(pkl, tmp_path / "w", name="demo")
    assert [p.name for p in paths] == ["demo_fold0.pt", "demo_fold1.pt"]
    payload = read_model_file(paths[1])                    # weights_only=True loads it
    assert payload["meta"]["fold"] == 1 and payload["meta"]["test_gauges"] == [11, 21]
    assert payload["norm"]["y_std"] == 12.5 and len(payload["norm"]["x_mean"]) == 12
    loaded, _ = load_model(paths[1])
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k
    # the bound buffers travel with the weights: tau floor 0.01 h in daily steps
    assert float(loaded.routing_model.offset) == pytest.approx(0.01 / 24)
    (tmp_path / "s").mkdir()
    single = export_run(_fake_run(tmp_path / "s", n_folds=1)[0], tmp_path / "w", name="one")
    assert [p.name for p in single] == ["one.pt"]


def test_check_norm(tmp_path):
    pkl, _ = _fake_run(tmp_path)
    payload = read_model_file(export_run(pkl, tmp_path / "w")[0])
    xm = torch.arange(12, dtype=torch.float64)
    check_norm(payload, xm, xm + 1, torch.tensor([12.5]))
    with pytest.raises(ValueError):
        check_norm(payload, xm, xm + 1, torch.tensor([13.0]))


def test_basin_groups_partition_whole_basins(synth):
    g = synth["g"].copy()
    g.add_edge(2000, 2001)                      # a second, two-node basin
    groups = basin_groups(g, 2)
    assert sorted(n for grp in groups for n in grp) == sorted(g.nodes)
    for grp in groups:
        assert (2000 in grp) == (2001 in grp)   # a basin is never split
    sub = basin_groups(g, 3, nodes=[1000, 2001])
    assert sorted(n for grp in sub for n in grp) == [1000, 2001]


def test_continuous_series_reproduces_training_evaluation(synth):
    """The released forward pass gives the series a training run reports."""
    t = synth["targets"]
    dss = init_split_dataset(synth["g"], synth["x"], synth["y"], synth["runoff_static"],
                             synth["routing_static"], synth["channel_length"], synth["area"],
                             t, t, t, INIT, PRED, irf_fn="instantaneous")
    model, _ = build_model("instantaneous", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    module = RRModule(model, *dss, scheduler_type="cosine", n_epoch=1, batch_size=1,
                      inference_batch_size=1, device="cpu", routing_lr=1e-4, routing_wd=0.0,
                      runoff_lr=1e-3, runoff_wd=0.0)
    _, ref = module.extract_test(device="cpu", batch_size=1)
    q, ro = continuous_series(model, dss[2], "cpu", batch_size=3, with_runoff=True)
    assert torch.allclose(q.values, ref.values, atol=1e-6)
    assert ro.shape == q.shape and torch.isfinite(ro.values).all()
    # whole-network simulation, in physical units, does not depend on the grouping
    data = (synth["g"], synth["x"], synth["y"], None, None, SimpleNamespace(values=torch.tensor(2.0)),
            synth["runoff_static"], synth["routing_static"], synth["channel_length"],
            synth["area"], None)
    spec = dict(init_window=INIT, pred_len=PRED, route_src_reach=False, irf_fn="instantaneous")
    times, Q1, _ = simulate(model, spec, data, synth["nodes"], "cpu", n_groups=1, log=lambda s: None)
    _, Q3, _ = simulate(model, spec, data, synth["nodes"], "cpu", n_groups=3, log=lambda s: None)
    assert Q1.shape == (len(synth["times"]) - INIT, len(synth["nodes"]))
    np.testing.assert_allclose(Q1, Q3, rtol=1e-5)
    cols = [synth["nodes"].index(n) for n in t]
    np.testing.assert_allclose(Q1[:, cols], ref.values.numpy().T * 2.0, rtol=1e-5, atol=1e-5)
    # a time slice keeps its warm-up before the first output day; windows that
    # start on the same day see the same warm-up and give the same values
    times_s, Qs, _ = simulate(model, spec, data, t, "cpu", time_slice=(str(times[100]), str(times[150])),
                              log=lambda s: None)
    assert times_s[0] == times[100] and times_s[-1] == times[150] and len(Qs) == 51
    np.testing.assert_allclose(Qs[:2 * PRED], Q1[100:100 + 2 * PRED][:, cols], rtol=1e-4, atol=1e-4)
