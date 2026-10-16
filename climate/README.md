# climate/ — climate-shift analysis (Jayadharshini)

Run from the repo root: `python -m climate.climate_impact` (add `--write-db` to also upsert into `climate_impact`).
Reads the change factors from the view `v_change_factors_region` if the database is up, else from
`data/golden/change_factors_region_monthly.csv`.

Method (delta-change): the weather-only model (wind, temperature, humidity + day-of-year) is trained on all days with
generation. Each climate model's 12 monthly factors are applied to the ERA5 record (wind x ratio, temperature + delta,
humidity + delta clipped to 0-100), the model predicts generation, and the change is compared with the unshifted
prediction. Spread across the 10 climate models = uncertainty. Shift diagnostic = share of days outside the 1st-99th
percentile of the training days.

Metrics in `climate_impact` (per climate model, plus `climate_model = 'ensemble'` with mean and p10/p90 across models):
`mean_generation_change_pct`, `..._wind_only`, `..._temperature_only`, `..._jun_sep`, `share_days_out_of_range`,
`share_days_temperature_out_of_range`; ensemble-only: `models_with_increase_pct`.
`climate_impact_sensitivity_linear.csv` repeats the run with a linear model; `climate_baseline_reference.csv` holds the
unshifted baseline (mean generation, out-of-range share with no shift).

Limits: today's fleet only (no capacity growth); 5 years of training generation; tree models cannot extrapolate beyond
what they saw; ERA5 10 m wind is low versus hub height.
