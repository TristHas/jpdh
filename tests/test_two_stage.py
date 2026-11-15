"""--runoff_init / --freeze_runoff: the frozen module leaves the optimizer,
the routing group keeps its learning rate, values are unchanged."""
import copy

import pytest
import torch

from diffhydro.pipelines.utils import collate_fn

from data import init_split_dataset
from training import RRModule, build_model
from conftest import ROUTING_FEATURES

PARAMS = dict(hidden_size=8, num_layers=1)


def _split(synth, irf="hayami"):
    t = synth["targets"]
    return init_split_dataset(synth["g"], synth["x"], synth["y"], synth["runoff_static"],
                              synth["routing_static"], synth["channel_length"],
                              synth["area"], t, t, t, 30, 20, irf_fn=irf)


def _module(model, dss, device="cpu"):
    return RRModule(model, *dss, scheduler_type="cosine", n_epoch=2, batch_size=1,
                    inference_batch_size=1, device=device, routing_lr=1e-4,
                    routing_wd=0.0, runoff_lr=1e-3, runoff_wd=0.0)


def test_unfrozen_model_has_both_groups(synth):
    model, _ = build_model("hayami", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    module = _module(model, _split(synth))
    groups = {g["name"]: g for g in module.opt.param_groups}
    assert set(groups) == {"runoff", "routing"}
    assert groups["runoff"]["lr"] == 1e-3 and groups["routing"]["lr"] == 1e-4


def test_frozen_runoff_leaves_only_the_routing_group(synth):
    model, _ = build_model("hayami", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    model.freeze_runoff()
    assert model.runoff_frozen
    assert all(not p.requires_grad for p in model.runoff_model.parameters())
    module = _module(model, _split(synth))
    groups = {g["name"]: g for g in module.opt.param_groups}
    assert set(groups) == {"routing"}
    assert groups["routing"]["lr"] == 1e-4
    routing_ids = {id(p) for p in model.routing_model.parameters()}
    assert {id(p) for p in groups["routing"]["params"]} == routing_ids


def test_frozen_and_parameterless_routing_is_refused(synth):
    model, _ = build_model("instantaneous", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    model.freeze_runoff()
    with pytest.raises(ValueError):
        _module(model, _split(synth, "instantaneous"))


def test_frozen_forward_matches_trainable_forward_cpu(synth):
    # the CPU inference LSTM kernel differs from the training one at float32
    # round-off; on cuDNN the two are bit-identical (asserted in the GPU test)
    dss = _split(synth, "instantaneous")
    model, _ = build_model("instantaneous", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    module = _module(model, dss)
    xb, _ = collate_fn([dss[0][0]])
    ref = module.run_model(dss[0], xb).values.detach().clone()
    assert ref.requires_grad is False and module.run_model(dss[0], xb).values.requires_grad
    model.freeze_runoff()
    out = module.run_model(dss[0], xb).values
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-6)
    assert not out.requires_grad          # nothing trainable feeds it


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DiffRoute kernels need CUDA")
def test_frozen_step_changes_only_routing_gpu(synth):
    dev = "cuda:0"
    dss = _split(synth, "hayami")
    model, _ = build_model("hayami", ROUTING_FEATURES, 3, 2, PARAMS, dev)
    module = _module(model, dss, device=dev)
    xb, _ = collate_fn([dss[0][0]])
    xb, stat = xb.to(dev), dss[0].statics["x_stat"].to(dev)
    ref_runoff = model.runoff_model(xb, stat).values.detach().clone()
    ref = module.run_model(dss[0], xb).values.detach().clone()
    model.freeze_runoff()
    module = _module(model, dss, device=dev)
    # same function: the LSTM matches to kernel round-off (cuDNN may pick an
    # inference kernel when no gradient is needed), the routed output to
    # round-off as well (DiffRoute's triton kernels use atomic adds)
    assert torch.allclose(model.runoff_model(xb, stat).values, ref_runoff,
                          rtol=1e-5, atol=1e-6)
    frozen = module.run_model(dss[0], xb).values
    assert torch.allclose(frozen, ref, rtol=1e-4, atol=1e-4)
    assert frozen.requires_grad                 # the routing parameters still train
    before_runoff = copy.deepcopy({k: v.clone() for k, v in model.runoff_model.state_dict().items()})
    before_routing = {k: v.clone() for k, v in model.routing_model.state_dict().items()
                      if v.is_floating_point()}
    module.train_epoch(n_iter=2, device=dev)
    for k, v in model.runoff_model.state_dict().items():
        assert torch.equal(v, before_runoff[k]), k
    changed = [k for k, v in model.routing_model.state_dict().items()
               if v.is_floating_point() and not torch.equal(v, before_routing[k])]
    assert changed, "the routing parameters did not move"


def test_init_output_bias_starts_at_mean_specific_runoff(synth):
    from model import mean_specific_runoff, init_output_bias
    from diffhydro.pipelines.utils import mm_to_m3s, collate_fn as cf
    from data import init_dataset
    g, area, y = synth["g"], synth["area"], synth["y"]
    target = mean_specific_runoff(y, g, area, synth["targets"], 24)
    assert target > 0
    model, _ = build_model("instantaneous", ROUTING_FEATURES, 3, 2, PARAMS, "cpu")
    init_output_bias(model.runoff_model, target)
    ds = init_dataset(g, synth["x"], y, synth["runoff_static"], synth["routing_static"],
                      synth["channel_length"], area, synth["targets"], 30, 20,
                      irf_fn="instantaneous")
    xb, _ = cf([ds[0]])
    out = model.runoff_model(xb, ds.statics["x_stat"]).values
    assert torch.allclose(out, torch.full_like(out, target), rtol=1e-4)
    # the routed discharge of a gauge then equals its mean target to within the
    # spread of the specific runoff across gauges (same order of magnitude)
    q = model(xb, ds.statics["x_stat"], ds.g, ds.statics["cat_area"]).values[0, :, 0]
    mean_q = torch.nanmean(y.sel(spatial=synth["targets"]).values[0], dim=-1)
    ratio = (q / mean_q)
    assert (ratio > 0.2).all() and (ratio < 5).all()
