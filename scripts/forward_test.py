import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from ml.model import GasPriceModel, HORIZONS
from ml.feature_engineering import FeatureEngineer


async def main():
    print("=" * 60)
    print("Проверка модели")
    print("=" * 60)

    data_path  = Path(__file__).parent.parent / 'data'
    model      = GasPriceModel()
    engineer   = FeatureEngineer()

    hist_files = sorted(data_path.glob('ethereum_historical_*.csv'), reverse=True)
    if not hist_files:
        print("No data files found")
        return

    df = pd.read_csv(hist_files[0])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    total     = len(df)
    test_start = int(total * 0.80)
    test_df   = df.iloc[test_start:]

    print(f"Total blocks:    {total}")
    print(f"Test blocks:     {len(test_df)} (20%)")
    print(f"Test date range: {test_df['datetime'].min()} to {test_df['datetime'].max()}")

    if 'ethereum' not in model.models:
        print("Model not loaded for ethereum")
        return

    test_engineered = engineer.create_all_features(test_df.copy())
    feature_cols    = model.feature_columns.get('ethereum', [])
    feature_cols    = [c for c in feature_cols if c in test_engineered.columns]

    if not feature_cols:
        print("No feature columns available")
        return

    X_test = test_engineered[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    print(f"\n{'Горизонт':<12} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10} {'Направление'}")
    print("-" * 65)

    for h, shift in HORIZONS.items():
        y_true = test_engineered['gas_price_gwei'].values
        y_pred = np.expm1(model.models['ethereum'][h].predict(X_test))

        valid  = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) < 2:
            continue

        mae    = float(np.mean(np.abs(y_true - y_pred)))
        rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2     = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        mask   = y_true != 0
        mape   = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

        dir_true = np.diff(y_true) > 0
        dir_pred = np.diff(y_pred) > 0
        dir_acc  = float(np.mean(dir_true == dir_pred) * 100)

        print(f"h={h}h{'':<8} {mae:<10.4f} {rmse:<10.4f} {r2:<10.4f} {mape:<10.2f} {dir_acc:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
