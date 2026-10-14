# AI use log

One line per session: tool | task | date

- Claude (claude.ai) | Phase 1: Docker + TimescaleDB setup, schema DDL, feature views (db/, docker-compose.yml) | 2026-10-09
- Claude (claude.ai) | Phase 2: CEA + ERA5 loaders, golden dataset export (etl/) | 2026-10-10
- Claude (claude.ai) | Phase 3: read_features() helper and tests (etl/features.py, tests/) | 2026-10-14
- Antigravity | Baseline models (persistence & linear regression), chronological validation, evaluation, plotting (models/baseline.py) | 2026-10-14
- Claude (Cowork) | Weather baselines (persistence, linear, horizons 1-3 days) and Random Forest/XGBoost with walk-forward validation (models/baseline_weather.py, models/trees.py); results independently re-run and checked for leakage | 2026-10-14
