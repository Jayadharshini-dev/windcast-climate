# Ramps and extremes (novelty extension)

Two modules answer a question that average error (RMSE) hides: **what happens on sudden changes and on very low or high generation days?**

| Command (repo root) | What it does | Writes |
| --- | --- | --- |
| `python -m models.ramps` | Scores every forecast model on real ramps (a k-day change at or above the 90th percentile of training-year changes), on low (<10th pct) and high (>90th pct) days, and gives the ramp calendar by month. Uses the stored holdout predictions; nothing is refitted. | `models/ramps_scores.csv`, `ramps_low_high_scores.csv`, `ramps_events_holdout.csv`, `ramps_by_month.csv`, `ramps_holdout.png` |
| `python -m climate.extremes_impact` | Runs the 10 climate models x 2 scenarios x 2 windows through the weather-only XGBoost and reports how often ramp days, low days, high days and low spells occur, against today. Thresholds are fixed on the baseline series. | `climate/extremes_impact.csv`, `extremes_baseline_reference.csv`, `extremes_impact.png` |
| `python -m climate.extremes_impact --model linear` | Same, with the linear model (sensitivity check). | files with a `_linear` suffix |

Limits: delta-change keeps today's day-to-day weather sequence, so the climate part shows the effect of a shifted mean on extremes, not of changed variability. Ramp scores use observed same-day weather and have only 23-26 real events per horizon in the holdout, so they are indicative. Tests: `python -m pytest tests/test_ramps.py`.

## Robustness checks (answers to the usual reviewer objections)

| Command | Objection it answers | Writes (`models/`) |
| --- | --- | --- |
| `python -m models.robustness --only noise` | Skill assumes perfect (observed) weather: test with degraded test weather, find the error level where the model stops beating persistence/climatology | `robustness_noise.csv`, `robustness_noise_summary.csv`, `robustness_noise_plain.png` |
| `python -m models.robustness --only noise_aware` | Remedy: train on observed + noisy copies of the weather | `robustness_noise_aware.csv`, `robustness_noise.png` |
| `python -m models.robustness --only baselines` | Weak baselines: power curve, spline (GAM-type), SARIMAX (needs `pip install statsmodels==0.14.4`; skipped if absent), Diebold-Mariano vs XGBoost | `robustness_baselines.csv` |
| `python -m models.robustness --only ramps` | Few ramps / one threshold: 80th, 90th, 95th percentile with block-bootstrap intervals | `robustness_ramp_sensitivity.csv` |
| `python -m models.robustness --only bias` | Capacity growth: mean error by half-year | `robustness_bias.csv` |
| `python -m climate.capacity_check` | Capacity growth: climate change re-estimated on generation rescaled to a fixed fleet (uses `data/golden/tn_wind_capacity_mw.csv`; the 2024 and 2025 rows are ASSUMED, replace them with reported values) | `climate/capacity_check.csv` |

Tests: `python -m pytest tests/test_robustness.py`.

## Daily climate-model output (variability and validation)

Needs an Earth Engine account (Pavan has one, or register free at https://code.earthengine.google.com/register and run `earthengine authenticate`).

1. `python -m etl.fetch_nexgddp_daily --ee-project YOUR_PROJECT --plan validation` (ssp245 2020-2025, 10 models, 60 calls)
2. `python -m etl.fetch_nexgddp_daily --ee-project YOUR_PROJECT --plan variability` (historical 1995-2014 + ssp585 2041-2060, 4 models, 160 calls). Files go to `data/raw/nexgddp_daily/` (git-ignored). Re-running skips finished files.
3. `python -m climate.daily_checks` runs both checks (`--validate` / `--variability` for one). Writes `climate/nexgddp_vs_era5.csv`, `nexgddp_vs_era5.png`, `variability_test.csv`.

Tests: `python -m pytest tests/test_daily_checks.py`.
