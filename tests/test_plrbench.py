import numpy as np

from plrbench import est, sim


def test_plant_shapes_and_truth():
    p = sim.make_plant(1)
    assert p["E"].shape == (sim.N_INV, sim.T) and p["true_plr"].shape == (sim.N_INV,)
    assert np.isnan(p["E"]).any() and (p["E"] == 0).any()


def test_yoy_recovers_plr_on_a_clean_plant():
    p = sim.make_plant(2)
    p["curt_flag"][:] = False
    plr, _ = est.yoy_plr(p, use_flags=True)
    assert abs(np.median(plr) - np.median(p["true_plr"])) < 0.5 + p["true_sensor_drift"] * 100  # sensor drift shifts everything


def test_flags_are_relative_to_the_fleet():
    plr = np.array([0.5, 0.5, 0.5, 1.5])
    assert list(est.flags(plr)) == [False, False, False, True]
    assert list(est.flags(plr + 3.0)) == [False, False, False, True]  # a plant-wide shift changes nothing


def test_known_swap_pairs_are_dropped():
    p = sim.make_plant(1000, held_out=True)
    i, day = next(iter(p["events"].items()))
    r = est.yoy_samples(p, False, True)
    straddle = np.arange(365, sim.T)
    bad = (straddle - 365 < day) & (straddle >= day)
    assert np.isnan(r[i][bad]).all()
