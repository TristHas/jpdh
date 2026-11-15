"""Per-gauge skill scores.

One function, ``per_gauge_metrics``, scores a table of simulated series
against the observed ones and returns one row per gauge.  Definitions (the
ones the paper states, and Yilmaz et al. 2008 for the flow-duration-curve
signatures):

* ``nse``      Nash-Sutcliffe efficiency.
* ``kge``      KGE' (Kling et al., 2012) ``1 - sqrt((r-1)^2 + (cv_ratio-1)^2 +
               (beta-1)^2)``.  The manuscript writes the correlation ``r`` as
               alpha and the coefficient-of-variation ratio ``cv_ratio``
               (CV_sim / CV_obs) as gamma; ``beta`` is mean_sim / mean_obs.
* ``nrmsd``    RMSE / mean(obs).
* ``fhv``      %BiasFHV: relative volume bias of the 2 % highest flows of each
               series' own flow-duration curve (Yilmaz eq. A3).
* ``flv``      %BiasFLV: log-domain bias of the 30 % lowest flows relative to
               each series' minimum, sign as in Yilmaz eq. A4 (positive = the
               simulated low-flow segment lies too high).
* ``fms``      %BiasFMS: bias of the log-slope of the FDC between the 20 % and
               70 % exceedance flows (Yilmaz eq. A2).
* ``lag``      the shift ``k`` in {-1, 0, +1} days maximising
               ``corr(sim(t + k), obs(t))``: +1 means the simulated hydrograph
               peaks one day *after* the observed one (late), -1 early.  Ties
               resolve to 0.
* ``nse_monthly``  NSE of the monthly means, over months with at least
               ``min_days_per_month`` scored days.
* ``n_valid``  number of scored days.

Every score uses only the days where both series are finite, intersected
with the optional ``mask`` (a temporal block); a gauge with fewer than
``min_valid_days`` scored days (364, one year of 12Z days) gets NaN
everywhere.  Percent signatures are in percent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_VALID_DAYS = 364   # one calendar year = 364 window-days in the 12Z convention
COLUMNS = ["nse", "kge", "r", "cv_ratio", "beta", "nrmsd",
           "fhv", "flv", "fms", "lag", "nse_monthly", "n_valid"]
_LOG_FLOOR = 1e-6   # flows at or below zero before taking logs (FDC signatures)


def _as_frame(a, times, name):
    if isinstance(a, pd.DataFrame):
        return a
    if times is None:
        raise ValueError(f"{name}: pass DataFrames, or arrays with `times`")
    return pd.DataFrame(np.asarray(a), index=pd.DatetimeIndex(times))


def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    denom = ((obs - obs.mean()) ** 2).sum()
    return 1.0 - ((sim - obs) ** 2).sum() / denom if denom > 0 else np.nan


def kge_components(obs, sim):
    """``(kge, r, cv_ratio, beta)``."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    mo, ms = obs.mean(), sim.mean()
    so, ss = obs.std(), sim.std()
    if so == 0 or ss == 0 or mo == 0 or ms == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    cv_ratio = (ss / ms) / (so / mo)
    beta = ms / mo
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (cv_ratio - 1) ** 2 + (beta - 1) ** 2)
    return kge, r, cv_ratio, beta


def nrmsd(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return np.sqrt(((sim - obs) ** 2).mean()) / obs.mean() if obs.mean() != 0 else np.nan


def fdc_signatures(obs, sim):
    """``(fhv, flv, fms)`` in percent, Yilmaz et al. (2008) A2-A4."""
    obs = np.sort(np.asarray(obs, float))[::-1]        # descending: exceedance
    sim = np.sort(np.asarray(sim, float))[::-1]
    n = len(obs)
    # A3 -- high segment: the H highest flows of each series (exceedance < 2 %)
    H = max(1, int(round(0.02 * n)))
    fhv = 100.0 * (sim[:H] - obs[:H]).sum() / obs[:H].sum() if obs[:H].sum() > 0 else np.nan
    # A4 -- low segment: the L lowest flows (exceedance 70..100 %), log domain
    L = max(1, int(round(0.30 * n)))
    lo_o = np.log(np.maximum(obs[n - L:], _LOG_FLOOR))
    lo_s = np.log(np.maximum(sim[n - L:], _LOG_FLOOR))
    vo = (lo_o - lo_o.min()).sum()
    vs = (lo_s - lo_s.min()).sum()
    flv = -100.0 * (vs - vo) / vo if vo > 0 else np.nan
    # A2 -- mid-segment slope between 20 % and 70 % exceedance
    qo1, qo2 = np.quantile(obs, 0.8), np.quantile(obs, 0.3)
    qs1, qs2 = np.quantile(sim, 0.8), np.quantile(sim, 0.3)
    so_ = np.log(max(qo1, _LOG_FLOOR)) - np.log(max(qo2, _LOG_FLOOR))
    ss_ = np.log(max(qs1, _LOG_FLOOR)) - np.log(max(qs2, _LOG_FLOOR))
    fms = 100.0 * (ss_ - so_) / so_ if so_ != 0 else np.nan
    return fhv, flv, fms


def best_lag(obs, sim, valid):
    """Shift ``k`` in {-1, 0, +1} maximising corr(sim(t+k), obs(t)); ties -> 0.

    ``obs``/``sim`` are the full (unfiltered) series and ``valid`` the boolean
    of scored days, so shifted pairs are formed on the original time axis.
    """
    def corr(k):
        if k >= 0:
            s, o, v = sim[k:], obs[:len(obs) - k], valid[k:] & valid[:len(obs) - k]
        else:
            s, o, v = sim[:k], obs[-k:], valid[:k] & valid[-k:]
        s, o = s[v], o[v]
        if len(s) < 2 or s.std() == 0 or o.std() == 0:
            return -np.inf
        return np.corrcoef(s, o)[0, 1]
    c0, c_plus, c_minus = corr(0), corr(1), corr(-1)
    if all(not np.isfinite(c) for c in (c0, c_plus, c_minus)):
        return np.nan
    # a shifted series has to beat the unshifted one by more than round-off
    # (a linear ramp correlates perfectly at every lag)
    best, lag = c0, 0
    for k, c in ((1, c_plus), (-1, c_minus)):
        if c > best + 1e-9:
            best, lag = c, k
    return lag


def monthly_nse(obs, sim, times, min_days_per_month=20):
    df = pd.DataFrame({"obs": obs, "sim": sim}, index=pd.DatetimeIndex(times))
    grp = df.groupby(df.index.to_period("M"))
    m = grp.mean()[grp.size() >= min_days_per_month]
    return nse(m["obs"].values, m["sim"].values) if len(m) >= 2 else np.nan


def per_gauge_metrics(obs, sim, times=None, mask=None,
                      min_valid_days=MIN_VALID_DAYS, min_days_per_month=20):
    """Score ``sim`` against ``obs`` gauge by gauge.

    ``obs``/``sim``: (time, gauge) DataFrames (the ``yte``/``ote`` tables of a
    result pickle, transposed as stored) or arrays with ``times``; ``mask``: a
    boolean over time (block); see the module docstring for the columns.
    """
    obs = _as_frame(obs, times, "obs")
    sim = _as_frame(sim, times, "sim")
    sim = sim.reindex(index=obs.index, columns=obs.columns)
    tidx = pd.DatetimeIndex(obs.index)
    if mask is None:
        mask = np.ones(len(tidx), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(tidx):
        raise ValueError("mask must be a boolean over the time axis")

    rows = []
    for col in obs.columns:
        o, s = obs[col].to_numpy(float), sim[col].to_numpy(float)
        valid = np.isfinite(o) & np.isfinite(s) & mask
        n = int(valid.sum())
        row = dict.fromkeys(COLUMNS, np.nan)
        row["n_valid"] = n
        if n >= min_valid_days:
            ov, sv = o[valid], s[valid]
            row["nse"] = nse(ov, sv)
            row["kge"], row["r"], row["cv_ratio"], row["beta"] = kge_components(ov, sv)
            row["nrmsd"] = nrmsd(ov, sv)
            row["fhv"], row["flv"], row["fms"] = fdc_signatures(ov, sv)
            row["lag"] = best_lag(o, s, valid)
            row["nse_monthly"] = monthly_nse(ov, sv, tidx[valid], min_days_per_month)
        rows.append(row)
    out = pd.DataFrame(rows, index=obs.columns, columns=COLUMNS)
    out.index.name = obs.columns.name or "spatial"
    return out


def summarize(table: pd.DataFrame, stat="median") -> pd.Series:
    """Column-wise median (or mean) over gauges, ignoring NaN; ``n`` = gauges scored."""
    numeric = table.drop(columns=["lag", "n_valid"], errors="ignore")
    out = getattr(numeric, stat)(skipna=True)
    if "lag" in table:
        lag = table["lag"].dropna()
        out["frac_lag_plus1"] = (lag == 1).mean() if len(lag) else np.nan
        out["frac_lag_minus1"] = (lag == -1).mean() if len(lag) else np.nan
    out["n"] = int(table["nse"].notna().sum())
    return out
