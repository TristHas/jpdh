"""Synthetic 20-node river tree shared by the tests (CPU, no dataset)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import networkx as nx  # noqa: E402
import numpy as np     # noqa: E402
import pandas as pd    # noqa: E402
import pytest          # noqa: E402
import torch           # noqa: E402
import xtensor as xt   # noqa: E402

N_NODES, N_DAYS = 20, 400
ROUTING_FEATURES = ["slope_mean", "upa_mean", "sinuosity"]
DYN_VARS = ["p0", "p1", "t0"]
STAT_VARS = ["s0", "s1"]
#: a block inside the series (June-September 2008, 122 days)
BLOCK = ("2008-06-01", "2008-09-30")


def make_synthetic(seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nodes = [1000 + i for i in range(N_NODES)]
    g = nx.DiGraph()
    for node in nodes:
        g.add_node(node, catchment_area=float(rng.uniform(5, 50)),
                   channel_length_km=float(rng.uniform(1, 10)))
    for i in range(1, N_NODES):                 # binary tree, outlet = nodes[0]
        g.add_edge(nodes[i], nodes[(i - 1) // 2])
    times = pd.date_range("2008-01-01 12:00", periods=N_DAYS, freq="D")
    x = xt.DataTensor(torch.randn(1, N_NODES, N_DAYS, len(DYN_VARS)),
                      dims=("batch", "spatial", "time", "variable"),
                      coords={"batch": xt.arange_index(1), "spatial": pd.Index(nodes),
                              "time": times, "variable": pd.Index(DYN_VARS)})
    targets = [nodes[0], nodes[1], nodes[5]]
    yv = torch.rand(1, len(targets), N_DAYS) + 0.5
    yv[0, 2, :50] = float("nan")                # a gap in one record
    y = xt.DataTensor(yv, dims=("batch", "spatial", "time"),
                      coords={"batch": xt.arange_index(1),
                              "spatial": pd.Index(targets), "time": times})
    df = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
    routing_static = pd.DataFrame(rng.standard_normal((N_NODES, len(ROUTING_FEATURES))),
                                  index=nodes, columns=ROUTING_FEATURES)
    runoff_static = xt.DataTensor(torch.randn(N_NODES, len(STAT_VARS)),
                                  dims=("spatial", "variable"),
                                  coords={"spatial": pd.Index(nodes),
                                          "variable": pd.Index(STAT_VARS)})
    return dict(g=g, nodes=nodes, x=x, y=y, targets=targets, times=times,
                channel_length=df["channel_length_km"], area=df["catchment_area"],
                routing_static=routing_static, runoff_static=runoff_static,
                block=BLOCK)


@pytest.fixture
def synth():
    return make_synthetic()


def upstream(g, node):
    return nx.ancestors(g, node) | {node}
