# plrbench: what makes per-inverter degradation estimates wrong?

Invertix's homepage shows an example analysis: an inverter degradation script that cleans the data (low irradiance, clipping, curtailment), splits an inverter at a warranty swap, fits a normal-behaviour model, computes a year-on-year performance index with a block-bootstrap interval, and flags inverters such as "INV-04 -1.23 %/yr [-1.41, -1.05]" against the fleet median (https://www.invertix.ai/). Every step of that example can go wrong in a way the final number hides. This prototype is a synthetic plant with known truth, to measure which steps matter.

## The setup
`plrbench/sim.py`: 40 inverters, 3 years of daily energy, known degradation per inverter (85% around 0.5 %/yr, 15% faulty at 1.2-2.5 %/yr). Shared nuisances: seasonal irradiance and temperature, soiling that builds and is rained off, curtailment days (only part flagged), an irradiance sensor that reads low over time. Per-inverter nuisances: stale-zero outages, missing days, warranty swaps that restart degradation.
`plrbench/est.py`: three estimators of each inverter's loss rate, then the flag rule "worse than the fleet median by more than 0.5 %/yr".
- **slope:** OLS slope of the daily performance index over all days.
- **yoy:** median year-on-year ratio of the index on days with enough irradiance (what most tools do).
- **yoy + flags + events + CI:** the same, skipping flagged curtailment days and index pairs that straddle a known swap, and requiring the 95% block-bootstrap (30-day blocks) lower bound of the gap to be above zero.

## Results (known truth; a "true flag" is an inverter whose true rate is more than 0.5 %/yr above the plant median)
Dev: 60 plants. Held-out: 100 plants (seeds 1000-1099) with heavier curtailment growth, 40% of curtailment days unflagged, a faster sensor drift (0.8 %/yr vs 0.2) and 2-4 warranty swaps, fixed in git before the first run.

| Held-out | precision | recall | false flags / missed | median abs error of the loss rate |
|---|---|---|---|---|
| slope over all days | 69.0% | 88.5% | 240 / 69 | 0.79 pts/yr |
| year-on-year | 94.9% | 92.7% | 30 / 44 | 0.54 |
| yoy + flags + swaps + CI | 93.7% | 95.7% | 39 / 26 | 0.67 |

Dev: slope 72.4% / 93.2%, yoy 95.9% / 98.9%, full 96.9% / 98.9%.

- Year-on-year is the big step: it removes soiling and seasonality, cutting false flags from 240 to 30 on held-out.
- The extra steps recover recall (swaps and flagged curtailment: 44 missed to 26) but cost precision (30 false flags to 39) and make the absolute rate slightly worse (0.54 to 0.67), because dropping data shrinks the sample. I did not tune this after the held-out run.
- **The absolute loss rate is not fixable from this data.** The sensor drift moves every inverter's estimate by about the same amount, so flags relative to the fleet survive and absolute numbers do not. The bootstrap interval covered the true rate 98.9% of the time on dev (nominal 95%) and only 76.7% on held-out, where the sensor drift is 0.8 %/yr: a reference cell is the only fix.

## Read this before believing the table
This uses a simple ratio index, not a trained normal-behaviour model, and my simulator, not real plants. Real swaps are not always logged, curtailment is more structured, and degradation is not linear. Nothing here uses Invertix data or code; the homepage example is a demonstration, not a measurement.

## Run it (3 commands)
```
pip install numpy pytest
python -m pytest -q tests
python -m plrbench.run
```
numpy only. 4 tests.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Invertix.
