"""--routing_bounds / --routing_init: the MLP router's bound and initial-value
buffers replaced from this repository's code, DiffHydro untouched."""
import numpy as np
import pytest
import torch

from diffhydro.pipelines.utils import PARAMS_BOUNDS

from model import INIT_LOGIT, override_router_buffers, parse_routing_bounds
from training import TEMP_RES_H, build_model
from conftest import ROUTING_FEATURES

PARAMS = dict(hidden_size=8, num_layers=1)


def test_parse_routing_bounds():
    assert parse_routing_bounds("tau=0,9.9") == {"tau": (0.0, 9.9)}
    assert parse_routing_bounds(" k=0.1,6 ; x=0,1 ") == {"k": (0.1, 6.0), "x": (0.0, 1.0)}
    assert parse_routing_bounds(None) == {} and parse_routing_bounds("") == {}


def test_released_buffers_are_diffhydro_s():
    model, _ = build_model("linear_storage", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    r = model.routing_model
    lo, hi = PARAMS_BOUNDS["linear_storage"]["tau"]["min"], PARAMS_BOUNDS["linear_storage"]["tau"]["max"]
    assert float(r.offset) == pytest.approx(lo / TEMP_RES_H)
    assert float(r.range) == pytest.approx((hi - lo) / TEMP_RES_H)
    assert float(r.param_init) == INIT_LOGIT["mid"]                     # released: mid-range


def test_override_floor_and_low_init():
    model, _ = build_model("linear_storage", ROUTING_FEATURES, 3, 2, PARAMS, "cpu",
                           routing_bounds={"tau": (0.0, 9.9)}, routing_init="low")
    r = model.routing_model
    assert float(r.offset) == 0.0
    assert float(r.range) == pytest.approx(9.9 / TEMP_RES_H)
    assert float(r.param_init) == INIT_LOGIT["low"]
    # the field starts near 4.7 % of the range and never below the floor
    with torch.no_grad():
        p = torch.zeros(50, len(ROUTING_FEATURES))
        f = r._read_params(p).numpy() * TEMP_RES_H                        # hours
    assert np.all(f >= 0) and abs(np.median(f) - 9.9 * 1 / (1 + np.exp(3))) < 0.2
    # the buffers travel with the state dict, so a readout sees the trained-under bounds
    sd = model.state_dict()
    assert float(sd["routing_model.offset"]) == 0.0 and float(sd["routing_model.param_init"]) == -3.0


def test_override_refuses_bad_input():
    model, _ = build_model("linear_storage", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    with pytest.raises(ValueError):
        override_router_buffers(model.routing_model, "linear_storage", TEMP_RES_H, bounds={"delay": (0, 1)})
    with pytest.raises(ValueError):
        override_router_buffers(model.routing_model, "linear_storage", TEMP_RES_H, bounds={"tau": (5, 1)})
    model, _ = build_model("instantaneous", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    with pytest.raises(ValueError):                   # a parameter-less router has no bounds
        override_router_buffers(model.routing_model, "linear_storage", TEMP_RES_H, init="low")
    # build_model leaves the instantaneous router alone even with an initial value given
    build_model("instantaneous", ROUTING_FEATURES, 3, 2, PARAMS, "cpu", routing_init="low")


def test_log_channel_length_descriptor(tmp_path):
    import pandas as pd
    from data import routing_static_table
    idx = pd.Index([1, 2, 3], name="node")
    pd.DataFrame({"log_upa": [1.0, 2.0, 3.0]}, index=idx).to_parquet(tmp_path / "routing_statics.parquet")
    nodes = pd.DataFrame({"channel_length_km": [1.0, np.e, np.e ** 2]}, index=idx)
    t = routing_static_table(tmp_path, nodes, ["log_channel_length_km", "log_upa"])
    raw = np.log(nodes["channel_length_km"]).values
    np.testing.assert_allclose(t["log_channel_length_km"].values, (raw - raw.mean()) / raw.std(ddof=1))
