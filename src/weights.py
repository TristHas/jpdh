"""Portable model files: a trained model's weights plus everything needed to rebuild it.

A model file (``.pt``, loadable with ``torch.load(..., weights_only=True)``) holds

* ``state_dict``  -- the coupled model's weights (runoff LSTM, routing MLP and
  the router's bound buffers);
* ``spec``        -- the configuration that rebuilds the model and its inputs:
  routing kernel, descriptors, bounds, initial value, runoff module size,
  dynamic / static input keys, window lengths;
* ``norm``        -- the normalisation constants of the training run (per
  forcing variable mean and std, and the discharge scale ``y_std``), checked
  against the ones recomputed from the dataset when the model is used;
* ``meta``        -- what the model was trained and tested on (fold, seed,
  training and test gauges, weight variant, library states).

``export_run`` writes one file per fold from a ``train.py`` result pickle made
with ``--save_weights``; ``load_model`` rebuilds the model on a device;
``run_payloads`` / ``model_from_payload`` do the same without writing files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import parse_routing_bounds
from training import build_model

SPEC_KEYS = ("irf_fn", "routing_features", "routing_bounds", "routing_init",
             "dynamic_keys", "static_runoff_keys", "runoff_hidden_size", "runoff_num_layers",
             "init_window", "pred_len", "route_src_reach")


def _floats(v) -> list[float]:
    if isinstance(v, (pd.Series, pd.DataFrame)):
        v = v.values
    elif not isinstance(v, (torch.Tensor, np.ndarray, list, tuple, float, int)) and hasattr(v, "values"):
        v = v.values                                    # xtensor DataTensor
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return [float(a) for a in np.asarray(v, dtype=np.float64).ravel()]


def _variables(x_mean) -> list[str]:
    try:
        return [str(v) for v in list(x_mean.coords["variable"])]
    except (AttributeError, KeyError, TypeError):
        return [str(v) for v in x_mean.index]


def run_payloads(pkl_path, name=None, variant="swa"):
    """The model-file payloads (one per fold) of a result pickle made with ``--save_weights``."""
    run = pd.read_pickle(pkl_path)
    cfg = run["config"]
    spec = {k: cfg[k] for k in SPEC_KEYS}
    spec["routing_features"] = list(spec["routing_features"])
    spec["dynamic_keys"] = list(spec["dynamic_keys"])
    spec["static_runoff_keys"] = list(spec["static_runoff_keys"])
    norm = {"variables": _variables(run["x_mean"]), "x_mean": _floats(run["x_mean"]),
            "x_std": _floats(run["x_std"]), "y_std": float(cfg["y_std"])}
    folds = run["fold_results"]
    name = name or Path(pkl_path).stem
    payloads = []
    for fr in folds:
        ev = fr[variant]
        if "weights" not in ev:
            raise ValueError(f"{pkl_path}: no weights for fold {fr['fold']} (train with --save_weights)")
        meta = {"fold": int(fr["fold"]), "seed": int(cfg["seed"]), "variant": variant,
                "name": name,
                "test_gauges": [int(g) for g in ev["nse_te"].index],
                "val_gauges": [int(g) for g in ev["nse_val"].index],
                "holdout_block": str(cfg.get("holdout_block", "none")),
                "libs": {str(k): str(v) for k, v in (cfg.get("libs") or {}).items()},
                "median_nse_test": float(np.nanmedian(ev["nse_te"].values))}
        payloads.append({"state_dict": {k: v.detach().cpu().clone() for k, v in ev["weights"].items()},
                         "spec": spec, "norm": norm, "meta": meta})
    return payloads


def export_run(pkl_path, out_dir, name=None, variant="swa"):
    """Write ``<out_dir>/<name>_fold<k>.pt`` for every fold of a result pickle
    (``<out_dir>/<name>.pt`` when the run has a single fold); returns the paths."""
    name = name or Path(pkl_path).stem
    payloads = run_payloads(pkl_path, name=name, variant=variant)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for payload in payloads:
        fold = payload["meta"]["fold"]
        path = out_dir / (f"{name}.pt" if len(payloads) == 1 else f"{name}_fold{fold}.pt")
        torch.save(payload, path)
        paths.append(path)
    return paths


def read_model_file(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(path, device="cpu", n_stat=None):
    """``(model, payload)``: the model of a model file, in eval mode on ``device``.

    ``n_stat`` (number of runoff static columns) is derived from the spec's
    static keys when not given.
    """
    payload = read_model_file(path)
    return model_from_payload(payload, device=device, n_stat=n_stat), payload


def model_from_payload(payload, device="cpu", n_stat=None):
    """The model of a payload (``read_model_file`` or ``run_payloads``), in eval mode on ``device``."""
    from config import expand_dynamic_keys, expand_static_keys
    spec = payload["spec"]
    if spec.get("routing_pspace", "linear") != "linear":
        raise ValueError(f"routing parameters in {spec['routing_pspace']} space are not supported")
    n_dyn = len(expand_dynamic_keys(spec["dynamic_keys"]))
    n_stat = n_stat or len(expand_static_keys(spec["static_runoff_keys"]))
    runoff_params = {"hidden_size": spec["runoff_hidden_size"],
                     "num_layers": spec["runoff_num_layers"]}
    model, _ = build_model(spec["irf_fn"], spec["routing_features"], n_dyn, n_stat,
                           runoff_params, device,
                           routing_bounds=parse_routing_bounds(spec["routing_bounds"]) or None,
                           routing_init=spec["routing_init"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def check_norm(payload, x_mean, x_std, y_std, rtol=1e-4, rtol_y=1e-3):
    """Fail when the dataset's normalisation constants differ from the model's:
    the model would then see differently scaled inputs than it was trained on.
    The discharge scale only has to be close (``rtol_y``): the model's outputs
    are converted with the scale stored in its model file, not the dataset's."""
    norm = payload["norm"]
    got = {"x_mean": _floats(x_mean), "x_std": _floats(x_std), "y_std": [float(_floats(y_std)[0])]}
    want = {"x_mean": norm["x_mean"], "x_std": norm["x_std"], "y_std": [norm["y_std"]]}
    for k in want:
        a, b = np.asarray(got[k]), np.asarray(want[k])
        if a.shape != b.shape or not np.allclose(a, b, rtol=rtol_y if k == "y_std" else rtol, atol=1e-6):
            raise ValueError(f"normalisation '{k}' of the dataset differs from the model's "
                             f"(max abs diff {np.max(np.abs(a - b)) if a.shape == b.shape else 'shape'}): "
                             f"is this the dataset the model was trained on?")
