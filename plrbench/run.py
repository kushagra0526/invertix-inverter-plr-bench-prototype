import json

import numpy as np

from . import est, sim


def score(seeds, held):
    tot = {k: dict(tp=0, fp=0, fn=0) for k in ("slope", "yoy", "yoy+flags+events+ci")}
    err = {"slope": [], "yoy": [], "yoy+flags+events+ci": []}
    cover = []
    for s in seeds:
        p = sim.make_plant(s, held)
        truth = (p["true_plr"] - np.median(p["true_plr"])) > 0.5
        rng = np.random.default_rng(s)
        a = est.slope_plr(p)
        b, _ = est.yoy_plr(p)
        c, reps = est.yoy_plr(p, use_flags=True, use_events=True, boot=150, rng=rng)
        for k, plr, rp in (("slope", a, None), ("yoy", b, None), ("yoy+flags+events+ci", c, reps)):
            f = est.flags(plr, rp)
            tot[k]["tp"] += int((f & truth).sum())
            tot[k]["fp"] += int((f & ~truth).sum())
            tot[k]["fn"] += int((~f & truth).sum())
            err[k].append(float(np.median(np.abs(plr - p["true_plr"]))))
        lo, hi = np.quantile(reps, [0.025, 0.975], axis=0)
        cover.append(float(((p["true_plr"] >= lo) & (p["true_plr"] <= hi)).mean()))
    out = {}
    for k, v in tot.items():
        out[k] = {"precision_pct": round(100 * v["tp"] / max(1, v["tp"] + v["fp"]), 1), "recall_pct": round(100 * v["tp"] / max(1, v["tp"] + v["fn"]), 1),
                  "false_flags": v["fp"], "missed": v["fn"], "median_abs_plr_error_pts": round(float(np.median(err[k])), 2)}
    out["ci_coverage_of_true_plr_pct"] = round(100 * float(np.mean(cover)), 1)
    return out


def main():
    res = {"dev": score(range(0, 60), False), "held_out": score(range(1000, 1100), True)}
    json.dump(res, open("out/result.json", "w"), indent=1)
    for k, v in res.items():
        print(k)
        for m, d in v.items():
            print("  ", m, d)


if __name__ == "__main__":
    main()
