"""
models/baseline.py

Baseline wind power forecasting models for Tamil Nadu, India.
Implements:
1. Persistence baseline (today's prediction = yesterday's value)
2. Linear Regression baseline (lags 1, 2, 3 and day-of-year sine/cosine harmonics)

Adheres strictly to PROJECT_CONTEXT.md rules:
- Daily grain only
- Features: lagged generation + calendar harmonics only
- Chronological time split (Train <= 2023-12-31, Test >= 2024-01-01)
- No imputation on target gaps (20 missing CEA days kept as gaps)
- Evaluation: RMSE, MAE, R2 on the test period
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Fix random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def run_baseline(
    data_path: str = "data/golden/cea_wind_daily_candidates.csv",
    output_plot_path: str = "models/baseline_actual_vs_predicted.png"
):
    print("=" * 70)
    print("WindCast Climate -- Baseline Forecasting Step for Tamil Nadu")
    print("=" * 70)

    # 1. Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    raw_df = pd.read_csv(data_path)
    df = raw_df[['date', 'Tamil Nadu']].copy()
    df.rename(columns={'Tamil Nadu': 'generation_mu'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"\n[Step 1] Loaded dataset: {data_path}")
    print(f"  Total records: {len(df)} days ({df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')})")
    
    # Missing data check
    missing_count = df['generation_mu'].isna().sum()
    print(f"  Missing days in Tamil Nadu wind generation: {missing_count} rows")
    if missing_count > 0:
        missing_dates = df[df['generation_mu'].isna()]['date'].dt.strftime('%Y-%m-%d').tolist()
        print(f"  Notice: Missing days are kept as gaps (NOT filled/interpolated).")
        print(f"  Missing dates sample: {missing_dates[:5]} ... ({len(missing_dates)} total)")

    # 2. Feature Engineering
    # Lagged generation: lag 1, 2, 3
    df['lag_1'] = df['generation_mu'].shift(1)
    df['lag_2'] = df['generation_mu'].shift(2)
    df['lag_3'] = df['generation_mu'].shift(3)

    # Calendar harmonics: day-of-year sine and cosine
    doy = df['date'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    feature_cols = ['lag_1', 'lag_2', 'lag_3', 'doy_sin', 'doy_cos']
    
    print("\n[Step 2] Feature Engineering")
    print("  Engineered features: lag_1 (t-1), lag_2 (t-2), lag_3 (t-3), doy_sin, doy_cos")

    # 3. Time-based Split (never random)
    split_date = pd.to_datetime("2023-12-31")
    train_mask = df['date'] <= split_date
    test_mask = df['date'] > split_date

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"\n[Step 3] Chronological Time-based Split")
    print(f"  Training period: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)} calendar days)")
    print(f"  Testing period:  {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')} ({len(test_df)} calendar days)")

    # Drop rows where target or needed lags are missing
    train_clean = train_df.dropna(subset=['generation_mu'] + feature_cols).copy()
    test_clean = test_df.dropna(subset=['generation_mu'] + feature_cols).copy()

    train_dropped = len(train_df) - len(train_clean)
    test_dropped = len(test_df) - len(test_clean)

    print(f"  Train: dropped {train_dropped} rows with missing lags/target -> {len(train_clean)} valid rows for fitting")
    print(f"  Test:  dropped {test_dropped} rows with missing lags/target -> {len(test_clean)} valid rows for evaluation")

    # 4. Models
    # --- Model A: Persistence Baseline ---
    # Prediction at day t is simply yesterday's value (lag_1)
    y_test = test_clean['generation_mu'].values
    y_pred_persist = test_clean['lag_1'].values

    rmse_persist = np.sqrt(mean_squared_error(y_test, y_pred_persist))
    mae_persist = mean_absolute_error(y_test, y_pred_persist)
    r2_persist = r2_score(y_test, y_pred_persist)

    # --- Model B: Linear Regression Baseline ---
    X_train = train_clean[feature_cols].values
    y_train = train_clean['generation_mu'].values
    X_test = test_clean[feature_cols].values

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print("\n[Step 4] Evaluation Metrics on Test Period (2024-01-01 onward)")
    print("-" * 65)
    print(f"{'Model':<25} | {'RMSE (MU/d)':<12} | {'MAE (MU/d)':<12} | {'R^2':<8}")
    print("-" * 65)
    print(f"{'Persistence (Lag 1)':<25} | {rmse_persist:<12.4f} | {mae_persist:<12.4f} | {r2_persist:<8.4f}")
    print(f"{'Linear Regression':<25} | {rmse_lr:<12.4f} | {mae_lr:<12.4f} | {r2_lr:<8.4f}")
    print("-" * 65)

    print("\nLinear Regression learned parameters:")
    print(f"  Intercept: {lr_model.intercept_:.4f}")
    for col, coef in zip(feature_cols, lr_model.coef_):
        print(f"  Coefficient for {col:<8}: {coef:+.4f}")

    # 5. Plotting Actual vs Predicted
    print(f"\n[Step 5] Generating actual vs predicted plot -> {output_plot_path}")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})

    test_dates = test_clean['date'].values

    # Top panel: Time series
    axes[0].plot(test_dates, y_test, label='Actual Generation', color='#1f77b4', linewidth=1.8, alpha=0.9)
    axes[0].plot(test_dates, y_pred_persist, label='Persistence Forecast (Lag 1)', color='#ff7f0e', linestyle='--', linewidth=1.2, alpha=0.7)
    axes[0].plot(test_dates, y_pred_lr, label='Linear Regression Forecast', color='#2ca02c', linestyle='-', linewidth=1.4, alpha=0.85)
    axes[0].set_title('Tamil Nadu Daily Wind Generation: Actual vs Baseline Forecasts (Test Set: 2024–2025)', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_ylabel('Generation (Million Units / Day)', fontsize=12)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right', frameon=True, fontsize=11)

    # Bottom panel: Prediction Errors (Residuals)
    persist_residuals = y_test - y_pred_persist
    lr_residuals = y_test - y_pred_lr

    axes[1].plot(test_dates, persist_residuals, label=f'Persistence Error (RMSE={rmse_persist:.2f})', color='#ff7f0e', alpha=0.5, linewidth=1.0)
    axes[1].plot(test_dates, lr_residuals, label=f'Linear Regression Error (RMSE={rmse_lr:.2f})', color='#2ca02c', alpha=0.7, linewidth=1.1)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    axes[1].set_title('Forecast Residuals (Actual - Predicted)', fontsize=12, pad=8)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Residual (MU / Day)', fontsize=12)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='lower left', frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"  Plot successfully saved to: {output_plot_path}")

    return {
        "rmse_persist": rmse_persist,
        "mae_persist": mae_persist,
        "r2_persist": r2_persist,
        "rmse_lr": rmse_lr,
        "mae_lr": mae_lr,
        "r2_lr": r2_lr,
        "lr_coefs": dict(zip(feature_cols, lr_model.coef_)),
        "lr_intercept": lr_model.intercept_,
    }

if __name__ == '__main__':
    run_baseline()
