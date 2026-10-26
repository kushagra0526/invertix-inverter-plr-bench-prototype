"""Per-inverter performance-loss-rate estimators and the flag rules built on them."""
import numpy as np

from .sim import T

POA_MIN = 2.0   # kWh/m2/day: skip dull days


def perf_index(p):
    return p["E"] / (p["poa_meas"] * p["f_t"])[None, :]


def slope_plr(p):
    """Naive: OLS slope of the daily index over all days, relative to its mean."""
    pi = perf_index(p)
    t = np.arange(T) / 365.25
    out = []
    for y in pi:
        ok = ~np.isnan(y)
        b = np.polyfit(t[ok], y[ok] / y[ok].mean(), 1)[0]
        out.append(-100 * b)
    return np.array(out)


def yoy_samples(p, use_flags, use_events):
    """Index ratios day d vs d-365 on days where both are usable. Returns (ratios [N, T-365], valid mask)."""
    pi = perf_index(p)
    ok = ~np.isnan(pi) & (pi > 0) & (p["poa_meas"] >= POA_MIN)[None, :]
    if use_flags:
        ok &= ~p["curt_flag"][None, :]
    pair = ok[:, 365:] & ok[:, :-365]
    if use_events:  # drop pairs that straddle a known swap
        for i, day in p["events"].items():
            idx = np.arange(365, T)
            pair[i] &= ~((idx - 365 < day) & (idx >= day))
    r = pi[:, 365:] / np.where(pi[:, :-365] == 0, np.nan, pi[:, :-365])
    return np.where(pair, r, np.nan)


def yoy_plr(p, use_flags=False, use_events=False, boot=0, rng=None):
    r = yoy_samples(p, use_flags, use_events)
    plr = -100 * (np.nanmedian(r, axis=1) - 1)
    if not boot:
        return plr, None
    n_blocks = r.shape[1] // 30
    reps = []
    for _ in range(boot):
        b = rng.integers(0, n_blocks, n_blocks)
        cols = np.concatenate([np.arange(k * 30, k * 30 + 30) for k in b])
        reps.append(-100 * (np.nanmedian(r[:, cols], axis=1) - 1))
    return plr, np.array(reps)           # (N,), (boot, N)


def flags(plr, reps=None, gap=0.5):
    """Flag inverters worse than the fleet median by more than `gap` %/yr; with bootstrap, also require the gap's lower bound > 0."""
    diff = plr - np.median(plr)
    f = diff > gap
    if reps is not None:
        d = reps - np.median(reps, axis=1, keepdims=True)
        f &= np.quantile(d, 0.025, axis=0) > 0
    return f
