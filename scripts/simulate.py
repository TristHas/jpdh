#!/usr/bin/env python3
"""Run a trained model over the river network: daily discharge at every catchment outlet.

    python scripts/simulate.py                                   # national simulation, shipped model
    python scripts/simulate.py --start 2019-01-01 --end 2019-12-31 --nodes gauges
    python scripts/simulate.py --model weights/full_data_seed00.pt --device cpu --n_groups 24

The model (default ``weights/full_data_seed00.pt``, the default configuration
trained on every training gauge over 2009-2020: the model of D3P-JP-Q) is run continuously, with
a 365-day warm-up before ``--start``, over the requested nodes, basin by basin
in ``--n_groups`` groups so that the routing kernel fits in memory.  Output
(``--out``, default ``results/simulation.zarr``):

* ``discharge(time, spatial)`` -- routed discharge at the catchment outlet, m3/s;
* ``runoff(time, spatial)``    -- the catchment's own runoff before routing,
  mm/day (with ``--runoff``);

``time`` = 12Z day labels (the day starting at 12:00 UTC), ``spatial`` = node
ids of ``network/nodes.parquet``.  A GPU with >= 24 GB runs the whole network
in about ten minutes; on a CPU, expect hours and raise ``--n_groups``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402

from config import data_root  # noqa: E402
from data import data_loading, read_splits_file, default_splits_path  # noqa: E402
from inference import simulate  # noqa: E402
from weights import check_norm, load_model  # noqa: E402

DEFAULT_MODEL = REPO / "weights" / "full_data_seed00.pt"


def resolve_nodes(spec, g, splits):
    if spec == "all":
        return list(g.nodes)
    if spec == "gauges":
        return [int(n) for n in splits["evaluation_set"]]
    return [int(v) for v in Path(spec).read_text().split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(REPO / "results" / "simulation.zarr"))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--start", default="2009-01-01", help="first output day (12Z label)")
    ap.add_argument("--end", default="2020-12-30", help="last output day (12Z label)")
    ap.add_argument("--nodes", default="all",
                    help="'all' (every catchment), 'gauges' (the evaluation gauges), or a "
                         "text file of node ids")
    ap.add_argument("--runoff", action="store_true", help="also write the catchments' own runoff")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_groups", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8, help="windows per forward pass")
    a = ap.parse_args()

    t0 = time.time()
    root = Path(a.data_root) if a.data_root else data_root()
    model, payload = load_model(a.model, device=a.device)
    spec, meta = payload["spec"], payload["meta"]
    print(f"model {a.model}: {spec['irf_fn']} routing on {spec['routing_features']}, "
          f"inputs {spec['dynamic_keys']} + {spec['static_runoff_keys']}", flush=True)
    data = data_loading(dynamic_keys=spec["dynamic_keys"],
                        static_runoff_keys=spec["static_runoff_keys"],
                        routing_static_var=spec["routing_features"], root=root)
    check_norm(payload, data[3], data[4], data[5])
    splits = read_splits_file(default_splits_path(root))
    nodes = resolve_nodes(a.nodes, data[0], splits)
    print(f"{len(nodes)} nodes, {a.start} .. {a.end} ({time.time() - t0:.0f} s to load)", flush=True)

    times, Q, R = simulate(model, spec, data, nodes, a.device, n_groups=a.n_groups,
                           batch_size=a.batch_size, with_runoff=a.runoff,
                           time_slice=(f"{a.start} 12:00", f"{a.end} 12:00"),
                           y_std=payload["norm"]["y_std"])
    if not np.isfinite(Q).all():
        raise RuntimeError("non-finite simulated discharge")

    variables = {"discharge": (("time", "spatial"), Q)}
    if a.runoff:
        variables["runoff"] = (("time", "spatial"), R)
    ds = xr.Dataset(variables, coords={"time": times.values,
                                       "spatial": np.asarray(nodes, np.int64)})
    ds.attrs = {
        "title": "Simulated daily river discharge on the D3P-JP river network",
        "model": Path(a.model).name,
        "model_spec": json.dumps(spec),
        "model_meta": json.dumps({k: meta[k] for k in ("name", "seed", "variant") if k in meta}),
        "time_convention": "12Z days: the label YYYY-MM-DD 12:00 UTC covers [12:00 UTC, next day 12:00 UTC)",
        "warm_up_days": int(spec["init_window"]),
    }
    ds["discharge"].attrs.update(units="m3 s-1", long_name="routed discharge at the catchment outlet")
    if a.runoff:
        ds["runoff"].attrs.update(units="mm day-1", long_name="catchment runoff before routing")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chunks = (len(times), min(256, len(nodes)))
    ds.to_zarr(out, mode="w", consolidated=True,
               encoding={v: {"chunks": chunks} for v in ds.data_vars})
    gpu = ""
    if torch.cuda.is_available() and str(a.device).startswith("cuda"):
        gpu = f", peak GPU memory {torch.cuda.max_memory_allocated(a.device) / 2**30:.1f} GiB"
    print(f"wrote {out}: {dict(ds.sizes)}, national mean of the node means "
          f"{float(Q.mean()):.2f} m3/s ({(time.time() - t0) / 60:.1f} min{gpu})", flush=True)


if __name__ == "__main__":
    main()
