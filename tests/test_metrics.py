"""src/metrics.py against hand-computed values."""
import numpy as np
import pandas as pd
import pytest

from metrics import per_gauge_metrics, summarize, COLUMNS


def _frame(values, start="2009-01-01 12:00"):
    values = np.asarray(values, float)
    return pd.DataFrame(values, index=pd.date_range(start, periods=len(values), freq="D"),
                        columns=["g"])


def test_hand_computed_example():
    obs = _frame([1, 2, 3, 4, 5])
    same = per_gauge_metrics(obs, obs, min_valid_days=1).loc["g"]
    for c in ("nse", "kge", "r", "cv_ratio", "beta"):
        assert same[c] == pytest.approx(1.0)
    for c in ("nrmsd", "fhv", "flv", "fms"):
        assert same[c] == pytest.approx(0.0, abs=1e-9)
    assert same["lag"] == 0 and same["n_valid"] == 5

    doubled = per_gauge_metrics(obs, obs * 2, min_valid_days=1).loc["g"]
    assert doubled["nse"] == pytest.approx(1 - 55 / 10)          # sum(o^2)=55, sum((o-3)^2)=10
    assert doubled["beta"] == pytest.approx(2.0)
    assert doubled["r"] == pytest.approx(1.0)
    assert doubled["cv_ratio"] == pytest.approx(1.0)
    assert doubled["kge"] == pytest.approx(0.0)                   # 1 - sqrt(0 + 0 + 1)
    assert doubled["nrmsd"] == pytest.approx(np.sqrt(55 / 5) / 3)
    assert doubled["fhv"] == pytest.approx(100.0)                 # H = 1: (10 - 5) / 5
    assert doubled["flv"] == pytest.approx(0.0, abs=1e-9)         # log-shape unchanged
    assert doubled["fms"] == pytest.approx(0.0, abs=1e-9)


def test_lag_mask_and_minimum_record():
    t = 600
    rng = np.random.default_rng(0)
    base = np.convolve(rng.random(t + 3), np.ones(3) / 3, mode="same")
    obs = _frame(base[:t])
    late = _frame(np.r_[np.nan, base[:t - 1]])          # simulation peaks one day later
    early = _frame(np.r_[base[1:t], np.nan])
    assert per_gauge_metrics(obs, late).loc["g", "lag"] == 1
    assert per_gauge_metrics(obs, early).loc["g", "lag"] == -1
    assert per_gauge_metrics(obs, obs).loc["g", "lag"] == 0
    # a time mask restricts the scored days; too few days -> NaN scores
    mask = np.zeros(t, bool); mask[:400] = True
    m = per_gauge_metrics(obs, obs, mask=mask).loc["g"]
    assert m["n_valid"] == 400 and m["nse"] == pytest.approx(1.0)
    mask[:] = False; mask[:100] = True
    short = per_gauge_metrics(obs, obs, mask=mask).loc["g"]
    assert short["n_valid"] == 100 and np.isnan(short["nse"])
    # monthly NSE exists and the summary carries the lag fractions
    tab = per_gauge_metrics(obs, late)
    assert list(tab.columns) == COLUMNS
    s = summarize(tab)
    assert s["frac_lag_plus1"] == 1.0 and s["n"] == 1
    assert np.isfinite(tab.loc["g", "nse_monthly"])
