"""Synthetic PV plant: 40 inverters, 3 years of daily energy, known per-inverter degradation (%/yr).

Common-mode nuisances every analyst meets: seasonal irradiance and temperature, soiling that builds and is rained off,
curtailment days (partly flagged), a drifting irradiance sensor. Per-inverter nuisances: stale-zero outages, missing days,
and swaps under warranty (degradation restarts from new).
"""
import numpy as np

N_INV, T = 40, 1095


def make_plant(seed, held_out=False):
    rng = np.random.default_rng(seed)
    d = np.arange(T)
    t = d / 365.25
    ar = np.zeros(T)
    for i in range(1, T):
        ar[i] = 0.6 * ar[i - 1] + rng.normal(0, 0.28)
    poa = np.clip((5.0 + 2.5 * np.sin(2 * np.pi * (d - 80) / 365.25)) * (1 + ar * 0.5), 0.3, None)
    amb = 15 + 10 * np.sin(2 * np.pi * (d - 110) / 365.25) + rng.normal(0, 2, T)
    f_t = 1 - 0.0035 * (amb + 4.0 * poa - 25)
    soil, s = np.zeros(T), 0.0
    for i in range(T):
        s = 0.0 if rng.random() < 0.035 else min(0.08, s + rng.uniform(0.0006, 0.0016))
        soil[i] = s
    faulty = rng.random(N_INV) < 0.15
    plr = np.where(faulty, rng.uniform(1.2, 2.5, N_INV), rng.normal(0.5, 0.15, N_INV))  # %/yr
    p_curt = 0.02 + (0.04 if held_out else 0.01) * t
    curt = rng.random(T) < p_curt
    curt_factor = np.where(curt, rng.uniform(0.5, 0.8, T), 1.0)
    flag_cov = 0.6 if held_out else 0.9
    curt_flag = curt & (rng.random(T) < flag_cov)
    sensor_drift = 0.008 if held_out else 0.002      # /yr, irradiance sensor reads low over time
    age = np.tile(t, (N_INV, 1))
    swaps = {}
    if held_out:
        for i in rng.choice(N_INV, int(rng.integers(2, 5)), replace=False):
            day = int(rng.integers(int(1.2 * 365), int(2.5 * 365)))
            swaps[int(i)] = day
            age[i, day:] = t[day:] - t[day]
    E = (poa * f_t * (1 - soil) * curt_factor)[None, :] * (1 - plr[:, None] / 100 * age) * rng.normal(1, 0.02, (N_INV, T))
    E[rng.random((N_INV, T)) < 0.01] = 0.0           # stale-zero outages
    E[rng.random((N_INV, T)) < 0.02] = np.nan         # missing days
    poa_meas = poa * (1 - sensor_drift * t)
    return {"E": E, "poa_meas": poa_meas, "f_t": f_t, "curt_flag": curt_flag, "events": swaps,
            "true_plr": plr, "faulty": faulty, "true_sensor_drift": sensor_drift}
