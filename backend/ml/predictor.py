import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from ml.model import GasPriceModel
from ml.feature_engineering import FeatureEngineer
from core.network_client import BlockchainClient

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 2, 4, 8, 12, 24]
RECENT_BLOCKS = 7200


class GasPricePredictor:

    def __init__(self) -> None:
        self.model = GasPriceModel()
        self.feature_engineer = FeatureEngineer()
        self.client = BlockchainClient()
        self._token_prices: Dict[str, float] = {"ethereum": 2000.0}
        self._ready: bool = False

    def initialize(self) -> bool:
        self._ready = len(self.model.models) > 0
        return self._ready

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError(
                "Predictor is not initialized or no trained models are loaded."
            )

    async def _refresh_token_price(self, network: str) -> None:
        try:
            price = await self.client.get_token_price_usd(network)
            if price and price > 0:
                self._token_prices[network] = float(price)
        except Exception as exc:
            logger.warning(
                "Failed to refresh token price for %s, using cached %.2f: %s",
                network, self._token_prices.get(network, 0), exc,
            )

    async def _fetch_features(self, network: str) -> pd.DataFrame:
        latest_block = await self.client.get_latest_block_number(network)
        blocks = await self.client.collect_block_range(network, latest_block, RECENT_BLOCKS)

        if not blocks:
            raise ValueError(f"No blocks returned for network '{network}'")

        df = pd.DataFrame(blocks)
        df["network"] = network
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["utilization"] = df["gas_used"] / df["gas_limit"]
        df["gas_price_gwei"] = df["base_fee_per_gas"] / 1e9

        engineered = self.feature_engineer.create_all_features(df)
        if engineered.empty:
            raise ValueError(f"Feature engineering produced empty DataFrame for '{network}'")

        return engineered

    async def get_current_gas_state(self, network: str = "ethereum") -> Dict:
        self._require_ready()
        df = await self._fetch_features(network)
        last = df.iloc[-1]
        return {
            "network": network,
            "gas_price_gwei": round(float(last.get("gas_price_gwei", 0)), 4),
            "utilization": round(float(last.get("utilization", 0)), 4),
            "datetime": str(last.get("datetime", "")),
        }

    async def predict_next_hours(
        self,
        network: str = "ethereum",
        horizons: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, float]]:
        self._require_ready()

        if network not in self.model.models:
            raise ValueError(f"No trained model for network '{network}'")

        if horizons is None:
            horizons = DEFAULT_HORIZONS

        df_features = await self._fetch_features(network)
        latest_row = df_features.iloc[[-1]]

        predictions: Dict[int, Dict[str, float]] = {}
        for h in horizons:
            try:
                result = self.model.predict_with_interval(network, latest_row, horizon=h)
                predictions[h] = {
                    "point": round(max(0.0, float(result["point"][0])), 4),
                    "lower": round(max(0.0, float(result["lower"][0])), 4),
                    "upper": round(max(0.0, float(result["upper"][0])), 4),
                }
            except Exception as exc:
                logger.error("Prediction failed for %s h=%dh: %s", network, h, exc)

        return predictions

    async def predict_optimal_deployment(
        self,
        gas_estimate: int,
        network: str = "ethereum",
        tx_per_day: int = 100,
        horizons: Optional[List[int]] = None,
    ) -> Dict:
        self._require_ready()

        await self._refresh_token_price(network)
        token_price = self._token_prices.get(network, 0.0)

        gas_oracle = await self.client.get_gas_oracle(network)
        current_gwei = float(gas_oracle.get("propose_gas_price", 0))
        current_cost_usd = self._calculate_cost(gas_estimate, current_gwei, token_price)

        predictions = await self.predict_next_hours(network=network, horizons=horizons)
        if not predictions:
            raise ValueError(f"No predictions available for network '{network}'")

        recommendations = []
        for h, pred in predictions.items():
            point_gwei = pred["point"]
            lower_gwei = pred["lower"]
            upper_gwei = pred["upper"]

            deployment_cost = self._calculate_cost(gas_estimate, point_gwei, token_price)
            daily_cost = self._calculate_cost(gas_estimate * tx_per_day, point_gwei, token_price)
            savings_pct = (
                (current_cost_usd - deployment_cost) / current_cost_usd * 100
                if current_cost_usd > 0 else 0.0
            )

            recommendations.append({
                "network": network,
                "deploy_in_hours": h,
                "predicted_gas_price_gwei": point_gwei,
                "predicted_lower_gwei": lower_gwei,
                "predicted_upper_gwei": upper_gwei,
                "deployment_cost_usd": round(deployment_cost, 4),
                "daily_operational_cost_usd": round(daily_cost, 2),
                "current_cost_usd": round(current_cost_usd, 4),
                "savings_vs_current_pct": round(savings_pct, 2),
            })

        best = min(recommendations, key=lambda x: x["deployment_cost_usd"])

        return {
            "all_options": sorted(recommendations, key=lambda x: x["deployment_cost_usd"]),
            "best_option": best,
            "current_gas_price_gwei": round(current_gwei, 4),
            "current_cost_usd": round(current_cost_usd, 4),
            "token_price_usd": token_price,
            "timestamp": datetime.now().isoformat(),
        }

    async def check_model_drift(self, network: str = "ethereum") -> Dict:
        self._require_ready()
        df = await self._fetch_features(network)
        return self.model.check_drift(network, df)

    @staticmethod
    def _calculate_cost(gas: int, price_gwei: float, token_price_usd: float) -> float:
        return gas * price_gwei * 1e-9 * token_price_usd
    
    @property
    def ready(self) -> bool:
        return self._ready
