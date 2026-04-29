import asyncio
import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.append(str(Path(__file__).parent.parent / 'backend'))

from ml.model import GasPriceModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


async def main():
    print("=" * 60)
    print("BLOCKCHAIN PERFORMANCE ANALYSIS - INITIAL SETUP")
    print("=" * 60)

    models_path = Path(__file__).parent.parent / 'backend' / 'models'
    models_path.mkdir(exist_ok=True)
    for old_model in models_path.glob('*_model.joblib'):
        old_model.unlink()
        print(f"Removed old model: {old_model.name}")

    data_path  = Path(__file__).parent.parent / 'data'
    hist_files = sorted(data_path.glob('ethereum_historical_*.csv'), reverse=True)

    if not hist_files:
        print("ERROR: No data files found. Run collect_smart_sampling.py first")
        return

    df = pd.read_csv(hist_files[0])
    print(f"Loaded {len(df)} records from {hist_files[0].name}")

    print(f"\nTRAINING MODEL")
    print(f"Total dataset: {len(df)} records")

    model   = GasPriceModel()
    metrics = model.train_all_networks(df)

    print("\nTraining Results:")
    print("-" * 60)
    for network, metrics_dict in metrics.items():
        print(f"{network.upper()}:")
        print(f"  MAE:  {metrics_dict['mae']:.4f}")
        print(f"  RMSE: {metrics_dict['rmse']:.4f}")
        print(f"  R²:   {metrics_dict['r2']:.4f}")
        print(f"  MAPE: {metrics_dict['mape']:.2f}%")

    print("=" * 60)
    print("Setup completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
    