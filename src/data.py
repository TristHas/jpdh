"""Dataset access: river network, PUB folds, temporal blocks, forcing and discharge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import torch
import xarray as xr
import xtensor as xt

import diffhydro as dh
import diffhydro.pipelines as dhp

from config import (
    DEFAULT_DYNAMIC_KEYS, DEFAULT_ROUTING_STATIC_VAR, DEFAULT_STATIC_RUNOFF_KEYS, data_root,
    expand_dynamic_keys, expand_static_keys,
)
from model import INSTANTANEOUS, attach_upstream_sum, riv_tree_irf

#: node attributes stored as parquet list columns, restored as python lists
_LIST_NODE_COLS = ("river_channel", "info")

#: forcing stores of the dataset, searched in this order (the second one is
#: the optional archive of the sensitivity experiments' inputs)
FORCING_STORES = ("forcing.zarr", "forcing_panels.zarr")


def dataset_root(root: Path | None = None) -> Path:
    """The dataset directory: ``root`` or ``config.data_root()``."""
    return Path(root) if root is not None else data_root()


def load_graph_data(root: Path | None = None, load_geometries: bool = False):
    """Load the river network graph and its node table.

    Returns ``(g, points)``; with ``load_geometries`` also the catchment
    polygons and channel lines.  ``g`` is a ``networkx.DiGraph`` (edges point
    downstream) whose node insertion order matches ``nodes.parquet``.  That
    order is load-bearing: it fixes the spatial ordering of every tensor built
    from ``g.nodes``.
    """
    net = dataset_root(root) / "network"

    nodes = pd.read_parquet(net / "nodes.parquet")
    edges = pd.read_parquet(net / "edges.parquet")

    g = nx.DiGraph()
    for node_id, attrs in zip(nodes.index, nodes.to_dict("records")):
        clean = {k: (list(v) if k in _LIST_NODE_COLS and isinstance(v, (list, tuple))
                     else v)
                 for k, v in attrs.items()}
        g.add_node(int(node_id), **clean)
    g.add_edges_from(zip(edges["source"].tolist(), edges["target"].tolist()))

    df = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat),
                              crs="epsg:4326")

    if load_geometries:
        return (g, points,
                gpd.read_parquet(net / "catchments.parquet"),
                gpd.read_parquet(net / "channels.parquet"))
    return g, points


# ---------------------------------------------------------------------------
# Splits (spatial folds) and temporal blocks
# ---------------------------------------------------------------------------

def default_splits_path(root: Path | None = None) -> Path:
    """The PUB folds shipped with the dataset (``splits/pub_folds_10.json``)."""
    return dataset_root(root) / "splits" / "pub_folds_10.json"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_splits_file(path) -> dict:
    """The raw payload of a splits file, plus ``_path`` / ``_sha256`` provenance.

    A splits file holds ``folds`` (a list of ``{train, val, test}`` node-id
    lists) and ``gauges_with_discharge`` (the set the global discharge
    normalisation is computed over); the dataset's file also lists the
    ``evaluation_set``, the ``training_pool`` and the ``time_blocks_file``,
    which ``time_blocks_path`` resolves.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    missing = {"folds", "gauges_with_discharge"} - payload.keys()
    if missing:
        raise ValueError(f"{path}: not a splits file, missing keys {sorted(missing)}")
    for i, f in enumerate(payload["folds"]):
        if not {"train", "val", "test"} <= f.keys():
            raise ValueError(f"{path}: fold {i} lacks train/val/test")
    payload["_path"] = str(path.resolve())
    payload["_sha256"] = sha256_file(path)
    return payload


def splits_from_payload(payload: dict):
    return [[set(f["train"]), set(f["val"]), set(f["test"])]
            for f in payload["folds"]]


def load_splits(root: Path | None = None, path=None):
    """The PUB folds: the dataset's own file, or ``path``.

    Returns ``(folds, gauges)`` where ``folds`` is a list of
    ``[train, val, test]`` node-id sets and ``gauges`` is the set the global
    discharge normalisation is computed over (``gauges_with_discharge``).
    """
    payload = read_splits_file(path or default_splits_path(root))
    return splits_from_payload(payload), payload["gauges_with_discharge"]


def time_blocks_path(splits_path, payload: dict | None = None, override=None):
    """Where the temporal blocks of a splits file live.

    Priority: an explicit ``override`` (``--time_blocks_file``); the splits
    file's ``time_blocks_file`` key, absolute or relative to the splits file;
    a ``time_blocks.json`` sitting next to the splits file.  ``None`` when
    nothing is found -- fine for runs without ``--holdout_block``.
    """
    splits_path = Path(splits_path)
    if override:
        return Path(override)
    key = (payload or {}).get("time_blocks_file")
    if key:
        p = Path(key)
        return p if p.is_absolute() else splits_path.parent / p
    sibling = splits_path.parent / "time_blocks.json"
    return sibling if sibling.exists() else None


def load_time_blocks(path) -> dict:
    """``{"B1": (Timestamp, Timestamp), ...}`` from ``time_blocks.json``.

    Every value that is a two-element list of dates is a block; other keys
    (descriptions) are ignored.  Bounds are inclusive calendar dates
    (B1 2009-2012, B2 2013-2016, B3 2017-2020).
    """
    payload = json.loads(Path(path).read_text())
    blocks = {}
    for k, v in payload.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            blocks[k] = (pd.Timestamp(v[0]), pd.Timestamp(v[1]))
    if not blocks:
        raise ValueError(f"{path}: no block found")
    return blocks


def time_index(dt: xt.DataTensor) -> pd.DatetimeIndex:
    """The ``time`` coordinate of a DataTensor as a DatetimeIndex."""
    t = dt.coords["time"]
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return pd.DatetimeIndex(t)


def block_mask(times, bounds) -> torch.Tensor:
    """Boolean over the time axis: True on the days of the block ``bounds``.

    The dataset labels its 12Z-12Z days at 12:00 UTC, so membership is decided
    on the calendar *date* of the label (inclusive on both ends), not on the
    timestamp -- ``2012-12-31T12:00`` belongs to a block ending 2012-12-31.
    """
    t = pd.DatetimeIndex(times).normalize()
    start, end = (pd.Timestamp(b).normalize() for b in bounds)
    return torch.from_numpy(np.asarray((t >= start) & (t <= end), dtype=bool))


def mask_targets(y: xt.DataTensor, holdout: torch.Tensor) -> xt.DataTensor:
    """A copy of ``y`` with the held-out days set to NaN (time is the last dim).

    This is what the *training* dataset is built from, so ``BaseDataset``
    computes its per-station ``y_var`` on training days only and the loss
    never sees a held-out target (NaN targets are masked out of the loss).
    Forcing is left untouched: it is an input, not a target.
    """
    holdout = torch.as_tensor(holdout, dtype=torch.bool)
    if y.dims[-1] != "time" or len(holdout) != y.shape[-1]:
        raise ValueError("mask_targets expects time as the last dim and a mask over it")
    values = y.values.clone()
    values[..., holdout.to(values.device)] = float("nan")
    return xt.DataTensor(values, dims=y.dims, coords=y.coords)


class TimeMaskedDataset(dhp.BaseDataset):
    """``BaseDataset`` that only samples windows whose prediction part lies
    inside the training blocks.

    ``train_days`` is a boolean over the time axis (True = a day the loss may
    see).  A start index ``idx`` is kept when every day of
    ``[idx + init_len, idx + init_len + pred_len)`` is a training day; the
    warm-up part may lie anywhere, forcing being an input and not a target.
    ``__len__``/``__getitem__`` map through that list, so the plain
    ``DataLoader(shuffle=True)`` of ``diffhydro.pipelines.base.init_optim_dl``
    samples uniformly among valid windows without knowing about blocks.

    Only for training: the inference sampler (``StridedStartSampler``) strides
    over ``range(len(ds))`` as time positions, which is wrong for a remapped
    index -- evaluation datasets stay plain ``BaseDataset`` over the full,
    unmasked series.
    """

    def __init__(self, x, y, init_len, pred_len, g=None, statics=None,
                 train_days=None):
        super().__init__(x, y, init_len, pred_len, g=g, statics=statics)
        n_time = len(self.x.coords["time"])
        n_start = n_time - self.total_len            # BaseDataset.__len__
        if train_days is None:
            train_days = torch.ones(n_time, dtype=torch.bool)
        train_days = torch.as_tensor(train_days, dtype=torch.bool).cpu()
        if len(train_days) != n_time:
            raise ValueError(f"train_days has {len(train_days)} entries for "
                             f"{n_time} time steps")
        # number of non-training days in each prediction part, via a prefix sum
        bad = torch.cat([torch.zeros(1, dtype=torch.int64),
                         (~train_days).to(torch.int64).cumsum(0)])
        starts = torch.arange(max(n_start, 0))
        lo = starts + self.init_len
        hi = lo + self.pred_len
        self.valid_starts = starts[(bad[hi] - bad[lo]) == 0]
        self.train_days = train_days
        if len(self.valid_starts) == 0:
            raise ValueError("no training window: the training blocks are "
                             "shorter than pred_len")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        return super().__getitem__(int(self.valid_starts[idx]))


def ancestors(g, n):
    return {n} | nx.ancestors(g, n)


def retreive_subg(g, nodes):
    return g.subgraph(set.union(*[ancestors(g, x) for x in nodes]))


def init_dataset(g, x, y,
                 runoff_static, routing_static,
                 channel_length, catchment_area,
                 target_nodes,
                 init_window, pred_len,
                 route_src_reach=False,
                 irf_fn="linear_storage",
                 train_days=None):
    g = retreive_subg(g, target_nodes)
    # The subgraph carries every ancestor of the targets, because they all feed
    # the routing, but only the targets are ever compared against observations.
    # `output_reach` drops the paths that end anywhere else, which shrinks both
    # the path closure and the routing kernel; the graph, the node ordering and
    # the runoff tensor the router expects are unaffected.
    g = dh.RivTree(g, irf_fn=riv_tree_irf(irf_fn),
                   param_df=routing_static,
                   route_src_reach=route_src_reach,
                   output_reach=list(target_nodes),
                   param_names=routing_static.columns)
    if irf_fn == INSTANTANEOUS:
        # the no-routing arm sums upstream runoff through this index pair
        # instead of DiffRoute's kernel (see model.InstantaneousRouter)
        attach_upstream_sum(g)
    statics = {
        "channel_dist": torch.from_numpy(channel_length[g.nodes].values).float(),
        "cat_area":     torch.from_numpy(catchment_area[g.nodes].values).float(),
        "x_stat":       runoff_static.sel(spatial=g.nodes).to(dtype=torch.float),
    }
    kwargs = dict(x=x.sel(spatial=g.nodes), y=y.sel(spatial=target_nodes),
                  g=g, init_len=init_window, pred_len=pred_len,
                  statics=statics)
    if train_days is None:
        return dhp.BaseDataset(**kwargs)
    return TimeMaskedDataset(train_days=train_days, **kwargs)


def init_split_dataset(g, x, y,
                       runoff_static, routing_static,
                       channel_length, catchment_area,
                       tr_nodes, val_nodes, te_nodes,
                       init_window, pred_len,
                       irf_fn="linear_storage",
                       route_src_reach=False,
                       y_train=None, train_days=None):
    """(train, val, test) datasets of one fold.

    ``y_train`` (targets with the held-out block set to NaN, see
    ``mask_targets``) and ``train_days`` apply to the training dataset only;
    val and test keep the full continuous series so the saved series can be
    scored per block afterwards.
    """
    def build(nodes, targets, days):
        return init_dataset(g, x, targets, runoff_static, routing_static,
                            channel_length, catchment_area, nodes,
                            init_window, pred_len,
                            irf_fn=irf_fn, route_src_reach=route_src_reach,
                            train_days=days)
    return (build(tr_nodes, y if y_train is None else y_train, train_days),
            build(val_nodes, y, None),
            build(te_nodes, y, None))


# ---------------------------------------------------------------------------
# Forcing stores
# ---------------------------------------------------------------------------

def forcing_stores(root: Path | None = None) -> list[Path]:
    """The dataset's forcing stores that exist."""
    daily = dataset_root(root) / "daily"
    own = [daily / s for s in FORCING_STORES if (daily / s).exists()]
    if not own:
        raise FileNotFoundError(f"no forcing store under {daily}: download the dataset first "
                                f"(python scripts/download_data.py)")
    return own


def open_forcing(stores, variables):
    """Load ``variables`` from the first store of ``stores`` that carries each.

    ``stores[0]`` is the dataset's ``daily/forcing.zarr``; the optional
    ``forcing_panels.zarr`` is searched next.  Every store must carry exactly
    the primary store's ``time`` and ``spatial`` coordinates -- they are
    checked, never aligned, because a shifted day label is precisely the kind
    of error that has to fail loudly.  Returns ``(xarray.Dataset,
    {variable: store})``; a variable found nowhere raises ``KeyError``.
    """
    stores = [Path(s) for s in stores]
    with xr.open_zarr(stores[0], consolidated=None) as primary:
        ref = {d: primary[d].values for d in ("time", "spatial")}
    remaining, pieces, origin = list(variables), [], {}
    for store in stores:
        with xr.open_zarr(store, consolidated=None) as ds:
            have = [v for v in remaining if v in ds.data_vars]
            if not have:
                continue
            sub = ds[have].load()
        for dim, values in ref.items():
            if dim not in sub.dims or len(sub[dim]) != len(values) \
                    or not np.array_equal(sub[dim].values, values):
                raise ValueError(f"{store}: '{dim}' coordinate differs from {stores[0]}")
        pieces.append(sub)
        origin.update({v: str(store) for v in have})
        remaining = [v for v in remaining if v not in have]
    if remaining:
        hint = ("" if any(s.name == "forcing_panels.zarr" for s in stores) else
                " -- the sensitivity experiments' inputs are in the optional panels archive "
                "(python scripts/download_data.py --panels)")
        raise KeyError(f"forcing variables {remaining} not found in any of "
                       f"{[str(s) for s in stores]}{hint}")
    return xr.merge(pieces, join="exact"), origin


#: routing descriptors read from ``network/nodes.parquet`` instead of
#: ``statics/routing_statics.parquet``
NODE_ROUTING_FEATURES = ("channel_length_km", "log_channel_length_km")
#: derived node descriptors: name -> (source column, transform)
NODE_DERIVED = {"log_channel_length_km": ("channel_length_km", np.log)}


def routing_static_table(statics_dir, node_table, routing_static_var):
    """The z-scored routing-descriptor table (node x descriptor) fed to the
    routing parameter module.

    Columns come from ``routing_statics.parquet``; ``channel_length_km`` -- the
    reach's channel length in km, untransformed -- comes from the node table,
    so that a kernel whose delay is expected to scale linearly with length can
    receive length itself rather than its log.  The stored log descriptors
    (``log_upa``, ``log_slope``) are already logged, so the order is always:
    transform, then z-score per column, then NaN (headwaters) -> 0.
    """
    rs = pd.read_parquet(statics_dir / "routing_statics.parquet")
    for col in routing_static_var:
        if col in NODE_DERIVED:
            src, fn = NODE_DERIVED[col]
            rs[col] = fn(node_table[src].reindex(rs.index).astype(float))
        elif col in NODE_ROUTING_FEATURES:
            rs[col] = node_table[col].reindex(rs.index).astype(float)
    missing = [c for c in routing_static_var if c not in rs.columns]
    if missing:
        raise KeyError(f"unknown routing descriptor(s) {missing}; "
                       f"available: {sorted(rs.columns)}")
    rs = rs[list(routing_static_var)]
    rs = (rs - rs.mean()) / rs.std()
    return rs.fillna(0)


def data_loading(dynamic_keys=None,
                 static_runoff_keys=None,
                 routing_static_var=None,
                 root=None,
                 splits_file=None):
    """Load everything a training or simulation run needs.

    Forcing and runoff statics are z-scored per variable over every node and
    day; discharge is divided by its global standard deviation over the gauges
    of the splits file's ``gauges_with_discharge``.  ``splits_file``: another
    partition than the dataset's (e.g. the single fold of a full-data run).

    Returns
    -------
    g, x, y, x_mean, x_std, y_std, runoff_statics, routing_statics,
    channel_length, area, splits
    """
    if dynamic_keys is None:       dynamic_keys = DEFAULT_DYNAMIC_KEYS
    if static_runoff_keys is None: static_runoff_keys = DEFAULT_STATIC_RUNOFF_KEYS
    if routing_static_var is None: routing_static_var = DEFAULT_ROUTING_STATIC_VAR

    dynamic_var = expand_dynamic_keys(dynamic_keys)
    runoff_static_var = expand_static_keys(static_runoff_keys)

    ds_root = dataset_root(root)
    data_dir = ds_root / "daily"
    statics_dir = ds_root / "statics"
    g, _ = load_graph_data(root=root)
    splits, gauges = load_splits(root=root, path=splits_file)

    # ---- routing statics (NaN on headwaters -> 0 after z-normalising) ----
    node_table = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient="index")
    routing_statics = routing_static_table(statics_dir, node_table, routing_static_var)

    # ---- runoff statics (raw physical values in the dataset, z-scored here) ----
    runoff_df = pd.read_parquet(statics_dir / "runoff_statics.parquet")[runoff_static_var]
    runoff_df = (runoff_df - runoff_df.mean()) / runoff_df.std()
    runoff_statics = xt.DataTensor(
        torch.from_numpy(runoff_df.values),
        dims=("spatial", "variable"),
        coords={"spatial": runoff_df.index, "variable": runoff_df.columns},
    ).sel(variable=runoff_static_var)
    runoff_statics = torch.nan_to_num(runoff_statics, 0)

    channel_length = node_table["channel_length_km"]   # km
    area = node_table["catchment_area"]                 # km2

    # ---- dynamic forcing ----
    dynamic_ds, _ = open_forcing(forcing_stores(root), dynamic_var)
    x = xt.Dataset.from_xarray(dynamic_ds)\
        .to_datatensor(dim="variable")\
        .expand_dims("batch")\
        .to(dtype=torch.float)\
        .sel(variable=dynamic_var)\
        .transpose("batch", "spatial", "time", "variable")

    # ---- discharge ----
    # Stored on the same spatial index as the forcing and the network, and
    # restricted to the gauge set the splits file declares.  Selected by name:
    # the store also carries the screening bitmask ``discharge_qc``.
    with xr.open_zarr(data_dir / "discharge.zarr", consolidated=None) as dsq:
        y = xt.DataTensor.from_dataarray(dsq["discharge"].load())
    y = y\
        .expand_dims("batch")\
        .to(dtype=torch.float)\
        .transpose("batch", "spatial", "time")\
        .sel(time=x["time"])\
        .sel(spatial=gauges)

    y_std = y.std(dim=("time", "spatial"))
    x_mean = x.mean(dim=("time", "spatial"))
    x_std = x.std(dim=("time", "spatial"))

    y = y / y_std
    x = torch.nan_to_num((x - x_mean) / x_std, nan=0.0)

    return (g, x, y, x_mean, x_std, y_std, runoff_statics, routing_statics,
            channel_length, area, splits)
