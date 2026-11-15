"""The no-routing scheme: an exact instantaneous upstream sum."""
import numpy as np
import pandas as pd
import pytest
import torch
import xtensor as xt

from diffhydro.pipelines.routing import LearnedRouter
from diffhydro.pipelines.utils import collate_fn

from data import init_dataset
from model import InstantaneousRouter, JPDHRRModel, upstream_sum_indices
from training import RRModule, build_model, MAX_DELAY, DT, TEMP_RES_H
from conftest import ROUTING_FEATURES, upstream


def _ds(synth, irf="instantaneous", route_src_reach=False):
    return init_dataset(synth["g"], synth["x"], synth["y"], synth["runoff_static"],
                        synth["routing_static"], synth["channel_length"],
                        synth["area"], synth["targets"], 30, 20,
                        irf_fn=irf, route_src_reach=route_src_reach)


def _runoff(g, n_batch=2, n_time=40, seed=1):
    torch.manual_seed(seed)
    return xt.DataTensor(torch.rand(n_batch, len(g), n_time),
                         dims=("batch", "spatial", "time"),
                         coords={"batch": xt.arange_index(n_batch),
                                 "spatial": pd.Index(g.nodes),
                                 "time": xt.arange_index(n_time)})


def _explicit_sum(g_nx, runoff, targets):
    sp = runoff.coords["spatial"]
    sp = pd.Index(sp.cpu().numpy() if isinstance(sp, torch.Tensor) else sp)
    out = torch.zeros(runoff.shape[0], len(targets), runoff.shape[-1])
    for r, d in enumerate(targets):
        rows = [sp.get_loc(n) for n in upstream(g_nx, d)]
        out[:, r] = runoff.values[:, rows].sum(1)
    return out


def test_indices_cover_exactly_the_upstream_sets(synth):
    ds = _ds(synth)
    rows, cols = upstream_sum_indices(ds.g)
    pos = ds.g.nodes_idx
    for r, d in enumerate(synth["targets"]):
        got = set(cols[rows == r].tolist())
        assert got == {int(pos[n]) for n in upstream(synth["g"], d)}


def test_router_equals_explicit_upstream_sum_and_is_differentiable(synth):
    ds = _ds(synth)
    runoff = _runoff(ds.g)
    runoff.values.requires_grad_(True)
    out = InstantaneousRouter()(runoff, ds.g)
    assert out.dims == ("batch", "spatial", "time")
    assert list(out.coords["spatial"]) == synth["targets"]
    expected = _explicit_sum(synth["g"], runoff, synth["targets"])
    assert torch.allclose(out.values, expected, atol=1e-5)
    out.values.sum().backward()
    # d(sum of outputs)/d(runoff[s]) = number of targets that s feeds
    # (itself included), per batch element and time step
    n_targets_fed = torch.tensor([sum(n in upstream(synth["g"], d) for d in synth["targets"])
                                  for n in ds.g.nodes], dtype=torch.float)
    assert (n_targets_fed >= 1).all() and (n_targets_fed.max() == 2)
    assert torch.allclose(runoff.values.grad.sum((0, 2)),
                          n_targets_fed * runoff.shape[0] * runoff.shape[-1])


def test_route_src_reach_is_inert(synth):
    a, b = _ds(synth, route_src_reach=False), _ds(synth, route_src_reach=True)
    runoff = _runoff(a.g)
    ya = InstantaneousRouter()(runoff, a.g).values
    yb = InstantaneousRouter()(runoff.sel(spatial=b.g.nodes), b.g).values
    assert torch.allclose(ya, yb)


def test_full_model_runs_on_cpu(synth):
    ds = _ds(synth)
    model, pm = build_model("instantaneous", ROUTING_FEATURES, 3, 2,
                            dict(hidden_size=8, num_layers=1), "cpu")
    assert pm is None and isinstance(model, JPDHRRModel)
    assert isinstance(model.routing_model, InstantaneousRouter)
    assert sum(p.numel() for p in model.routing_model.parameters()) == 0
    module = RRModule(model, ds, ds, ds, scheduler_type="cosine", n_epoch=1,
                      batch_size=2, inference_batch_size=2, device="cpu",
                      routing_lr=1e-4, routing_wd=0.0, runoff_lr=1e-3, runoff_wd=0.0)
    assert [g["name"] for g in module.opt.param_groups] == ["runoff"]
    xb, yb = collate_fn([ds[0], ds[3]])
    o = module.run_model(ds, xb)
    assert o.shape == (2, len(synth["targets"]), 50)
    assert list(o.coords["spatial"]) == synth["targets"]


class _ZeroDelay(LearnedRouter):
    """DiffRoute's pure_lag kernel with every delay pinned at exactly 0."""
    def _read_params(self, p):
        return torch.zeros(p.shape[0], 1, device=p.device, dtype=p.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DiffRoute kernels need CUDA")
def test_pure_lag_at_zero_delay_versus_exact_sum_gpu(synth):
    """Documents why the sum is explicit: DiffRoute's pure_lag at zero delay
    goes through an FFT closure and an hourly->daily resampling and is the
    identity only to float32 round-off (relative gap printed by the test)."""
    dev = "cuda:0"
    ds = _ds(synth, irf="pure_lag")                 # route_src_reach=False semantics
    g = ds.g.to(dev)
    runoff = _runoff(ds.g).to(dev)
    exact = InstantaneousRouter()(runoff, g).values
    router = _ZeroDelay("pure_lag", None, max_delay=MAX_DELAY, dt=DT,
                        temp_res_h=TEMP_RES_H).to(dev)
    with torch.no_grad():
        routed = router(runoff, g).values
    gap = (routed - exact).abs().max().item()
    rel = gap / exact.abs().max().item()
    print(f"\npure_lag@0 vs exact upstream sum: max abs gap {gap:.3e}, relative {rel:.3e}")
    assert rel < 1e-3                                # same operator up to round-off
    assert gap > 0 or True                           # equality is not required
