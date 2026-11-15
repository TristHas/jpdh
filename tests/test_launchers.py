"""experiments/: job construction, pass-through flags and the configuration tables."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import _common
from config import KERNEL_BOUNDS


def _ns(**kw):
    base = dict(device="cuda:3", extra=[], shard=None, force=False, dry_run=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_flags_dict():
    assert _common.flags_dict(["--save_series", "--dynamic_vars", "a,b", "--x"]) == \
        {"--save_series": True, "--dynamic_vars": "a,b", "--x": True}
    # a configuration's own flag comes after the base and overrides it
    assert _common.flags_dict(["--irf_fn", "a", "--irf_fn", "b"])["--irf_fn"] == "b"


def test_job_command(tmp_path):
    a = _ns(extra=["--max_folds", "1"])
    out, cmd = _common.job(a, "pub", 3, tmp_path, ["--irf_fn", "instantaneous"])
    assert out == tmp_path / "03_pub.pkl"
    assert cmd[1:3] == ["-u", str(_common.REPO / "train.py")]
    assert cmd[cmd.index("--seed") + 1] == "3"
    assert cmd[cmd.index("--device") + 1] == "cuda:3"
    assert cmd[-4:] == ["--irf_fn", "instantaneous", "--max_folds", "1"]


def test_pending_filters_done_and_shards(tmp_path):
    jobs = [(tmp_path / f"{i:02d}_x.pkl", ["c", str(i)]) for i in range(6)]
    jobs[1][0].touch()
    assert [c[1] for _, c in _common.pending(_ns(), jobs)] == ["0", "2", "3", "4", "5"]
    assert [c[1] for _, c in _common.pending(_ns(shard="1/2"), jobs)] == ["3", "5"]
    assert len(_common.pending(_ns(force=True), jobs)) == 6


def test_parse_args_passthrough(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--seeds", "0-1", "--dry_run", "--",
                                      "--max_folds", "1", "--save_series"])
    a = _common.parse_args("d")
    assert a.seeds == "0-1" and a.dry_run
    assert a.extra == ["--max_folds", "1", "--save_series"]
    assert list(_common.seed_range("3-5")) == [3, 4, 5]


def test_check_refuses_typos():
    _common.check([("ok", ["--dynamic_vars", "gauge_prcp_4h_bin,msm_a_temp_4h_bin"])])
    with pytest.raises(KeyError):
        _common.check([("bad", ["--dynamic_vars", "gauge_prcp_4h_bins"])])
    with pytest.raises(ValueError):
        _common.check([("bad", ["--irf_fn", "hayamy"])])


def test_experiment_tables():
    import dynamic_inputs, precipitation, routing_scheme, runoff_statics, \
        temporal_resolution  # noqa: E401
    for mod in (dynamic_inputs, precipitation, routing_scheme, runoff_statics,
                temporal_resolution):
        _common.check(mod.CONFIGS)
        assert len({n for n, _ in mod.CONFIGS}) == len(mod.CONFIGS)
    # the configurations drawn in Figure 9, panels a-e (the default configuration is the pub cell)
    assert {n for n, _ in precipitation.CONFIGS} == {f"prcp_{p}" for p in ("garadar", "radar", "msm", "era")}
    assert {n for n, _ in temporal_resolution.CONFIGS} == {f"res_{r}" for r in ("1h", "2h", "6h", "12h")}
    assert {n for n, _ in dynamic_inputs.CONFIGS} == {f"extra_{v}" for v in ("wind", "sp", "ncld", "all")}
    assert len(runoff_statics.CONFIGS) == 13
    names = dict(routing_scheme.CONFIGS)
    assert set(names) == {"irf_nash_cascade_L", "irf_muskingum_L", "irf_pure_lag_L", "irf_hayami_SA"}
    for name, flags in routing_scheme.CONFIGS:
        f = _common.flags_dict(flags)
        assert f["--routing_bounds"] == KERNEL_BOUNDS[f["--irf_fn"]]
        assert ("channel_length_km" in f["--routing_static_vars"]) == (f["--irf_fn"] != "hayami")
    assert _common.flags_dict(names["irf_pure_lag_L"])["--routing_bounds"] == "delay=0,0.5"


def test_setting_cells():
    import settings
    assert [n for n, _ in settings.cell_configs("pubt")] == ["pubt_B1", "pubt_B2", "pubt_B3"]
    for cell in settings.CELLS:
        for name, flags in settings.cell_configs(cell):
            f = _common.flags_dict(flags)
            assert "--save_series" in f and "--save_weights" in f
            assert ("--holdout_block" in f) == (cell == "pubt")
    with pytest.raises(ValueError):
        settings.cell_configs("unknown")


def test_training_recipe_arms():
    import training_recipe
    two, no = (_common.flags_dict(training_recipe.CONFIGS[k][1]) for k in ("two_stage", "noroute"))
    assert "--freeze_runoff" in two and "{seed:02d}_lumped_fold{fold}.pt" in two["--runoff_init"]
    assert no["--irf_fn"] == "instantaneous" and "--freeze_runoff" not in no


def test_dry_runs_list_commands():
    import subprocess
    exp = Path(_common.__file__).parent
    for script, extra, n in [("settings.py", ["--cells", "pub,pubt"], 4),
                             ("training_recipe.py", ["--arms", "lumped,two_stage,noroute"], 3),
                             ("routing_scheme.py", [], 4),
                             ("precipitation.py", [], 4),
                             ("temporal_resolution.py", [], 4),
                             ("dynamic_inputs.py", [], 4),
                             ("runoff_statics.py", [], 13)]:
        out = subprocess.run([sys.executable, str(exp / script), *extra, "--seeds", "0-0",
                              "--dry_run", "--force"], check=True, capture_output=True,
                             text=True).stdout
        cmds = [line for line in out.splitlines() if line.startswith(sys.executable)]
        assert len(cmds) == n, (script, out)


def test_full_data_split(tmp_path):
    from full_data import derive_fulldata_splits
    src = tmp_path / "pub_folds_10.json"
    src.write_text(json.dumps({"folds": [{"train": [1], "val": [2], "test": [2, 3]}],
                               "gauges_with_discharge": [1, 2, 3], "training_pool": [1, 2],
                               "evaluation_set": [1, 2, 3], "basin_folds": {"9": 0}}))
    dst = derive_fulldata_splits(src, tmp_path / "full.json")
    d = json.loads(dst.read_text())
    assert d["folds"] == [{"fold": 0, "train": [1, 2], "val": [1, 2], "test": [1, 2, 3]}]
    assert d["gauges_with_discharge"] == [1, 2, 3] and "basin_folds" not in d
    assert derive_fulldata_splits(src, dst) == dst                   # idempotent
