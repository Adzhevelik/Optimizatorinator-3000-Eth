import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

HORIZONS = {h: h for h in range(1, 25)}


class GasPriceModel:

    def __init__(self):
        self.models: Dict[str, Dict[int, Any]] = {}
        self.quantile_models: Dict[str, Dict[int, Tuple[Any, Any]]] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.log_price_last: Dict[str, float] = {}
        self.feature_engineer = FeatureEngineer()

        self.model_path = Path(__file__).resolve().parent.parent.parent / "models"
        self.model_path.mkdir(exist_ok=True)

        self.network_block_times = {
            "ethereum": 12,
            "polygon": 2,
            "bsc": 3,
            "arbitrum": 1,
            "optimism": 2,
        }

        self.test_hours = 24
        self.halflife_days = 30
        self.fresh_weight = 2.0
        self.historical_weight = 0.3

        self.base_params = {
            "ethereum": {"max_bins": 255, "random_state": 42},
            "default":  {"max_bins": 255, "random_state": 42},
        }

        self.horizon_params = {
            "ethereum": {
                "short": {
                    "loss": "absolute_error",
                    "max_iter": 1000,
                    "learning_rate": 0.005,
                    "max_depth": 4,
                    "min_samples_leaf": 15,
                    "l2_regularization": 0.5,
                    "max_leaf_nodes": 31,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                    "n_iter_no_change": 30,
                },
                "medium": {
                    "loss": "absolute_error",
                    "max_iter": 700,
                    "learning_rate": 0.003,
                    "max_depth": 3,
                    "min_samples_leaf": 40,
                    "l2_regularization": 1.0,
                    "max_leaf_nodes": 24,
                    "early_stopping": True,
                    "validation_fraction": 0.15,
                    "n_iter_no_change": 40,
                },
                "long": {
                    "loss": "absolute_error",
                    "max_iter": 500,
                    "learning_rate": 0.002,
                    "max_depth": 2,
                    "min_samples_leaf": 60,
                    "l2_regularization": 2.0,
                    "max_leaf_nodes": 15,
                    "early_stopping": True,
                    "validation_fraction": 0.2,
                    "n_iter_no_change": 50,
                },
            },
            "default": {
                "short": {
                    "loss": "absolute_error",
                    "max_iter": 1000,
                    "learning_rate": 0.005,
                    "max_depth": 4,
                    "min_samples_leaf": 15,
                    "l2_regularization": 0.5,
                    "max_leaf_nodes": 31,
                    "early_stopping": True,
                    "validation_fraction": 0.1,
                    "n_iter_no_change": 30,
                },
                "medium": {
                    "loss": "absolute_error",
                    "max_iter": 700,
                    "learning_rate": 0.003,
                    "max_depth": 3,
                    "min_samples_leaf": 40,
                    "l2_regularization": 1.0,
                    "max_leaf_nodes": 24,
                    "early_stopping": True,
                    "validation_fraction": 0.15,
                    "n_iter_no_change": 40,
                },
                "long": {
                    "loss": "absolute_error",
                    "max_iter": 500,
                    "learning_rate": 0.002,
                    "max_depth": 2,
                    "min_samples_leaf": 60,
                    "l2_regularization": 2.0,
                    "max_leaf_nodes": 15,
                    "early_stopping": True,
                    "validation_fraction": 0.2,
                    "n_iter_no_change": 50,
                },
            },
        }

        self._load_all_models()

    def _detect_block_time(self, data: pd.DataFrame, network: str) -> float:
        default = self.network_block_times.get(network, 12)
        if "datetime" not in data.columns or len(data) < 10:
            return default
        dt = pd.to_datetime(data["datetime"]).sort_values().reset_index(drop=True)
        intervals = dt.diff().dt.total_seconds().dropna()
        median_interval = intervals.median()
        if 1.0 < median_interval < 600.0:
            logger.info(
                f"[{network}] Detected block_time={median_interval:.1f}s "
                f"from data (default={default}s)"
            )
            return float(median_interval)
        return default

    def _get_params_for_horizon(self, network: str, h: int) -> dict:
        base = self.base_params.get(network, self.base_params["default"])
        zones = self.horizon_params.get(network, self.horizon_params["default"])
        zone = "short" if h <= 8 else "medium" if h <= 16 else "long"
        return {**base, **zones[zone]}

    def _get_quantile_params(self, network: str, h: int, quantile: float) -> dict:
        params = self._get_params_for_horizon(network, h).copy()
        params["loss"] = "quantile"
        params["quantile"] = quantile
        params["early_stopping"] = False
        params.pop("n_iter_no_change", None)
        params.pop("validation_fraction", None)
        params["max_iter"] = max(200, params.get("max_iter", 500) // 3)
        return params

    def _load_all_models(self):
        self._load_model("ethereum")

    def _load_model(self, network: str):
        model_file = self.model_path / f"{network}_model.joblib"
        if model_file.exists():
            try:
                saved = joblib.load(model_file)
                self.models[network] = saved["models"]
                self.feature_columns[network] = saved["feature_columns"]
                self.log_price_last[network] = saved.get("log_price_last", {})
                if "quantile_models" in saved:
                    self.quantile_models[network] = saved["quantile_models"]
                logger.info(f"Model loaded for {network}")
            except Exception as e:
                logger.warning(f"Failed to load model for {network}: {e}")

    def _save_model(self, network: str):
        model_file = self.model_path / f"{network}_model.joblib"
        joblib.dump(
            {
                "models": self.models[network],
                "feature_columns": self.feature_columns[network],
                "log_price_last": self.log_price_last.get(network, {}),
                "quantile_models": self.quantile_models.get(network, {}),
            },
            model_file,
        )
        logger.info(f"Model saved for {network}")

    def _prepare_features(self, data: pd.DataFrame):
        exclude_cols = {
            "network", "timestamp", "datetime", "gas_price_gwei",
            "number", "base_fee_per_gas", "source",
        }
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        X = data[feature_cols].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        return X, feature_cols

    def _calculate_weights(self, data: pd.DataFrame, size: int) -> Optional[np.ndarray]:
        if "datetime" not in data.columns:
            return None
        dates = pd.to_datetime(data["datetime"], errors="coerce")
        ref_date = dates.max()
        days_old = (ref_date - dates).dt.total_seconds() / 86400
        days_old = days_old.fillna(30).clip(0, 365)
        weights = np.exp(-np.log(2) * days_old / self.halflife_days)
        if "source" in data.columns:
            weights[data["source"].str.contains("fresh", na=False)] *= self.fresh_weight
            weights[data["source"].str.contains("kaggle", na=False)] *= self.historical_weight
        return weights.iloc[:size].values

    def _naive_baseline_mae(self, y_true: np.ndarray) -> float:
        if len(y_true) < 2:
            return float("nan")
        return float(np.mean(np.abs(y_true[1:] - y_true[:-1])))

    def _try_build_gnn_features(
        self, X: pd.DataFrame, y_log: pd.Series, network: str
    ) -> Optional[pd.DataFrame]:
        try:
            import torch
            from ml.graph_model import GraphFeatureExtractor
        except ImportError:
            logger.info(f"[{network}] torch/graph_model not available, skipping GNN")
            return None

        if len(X) < 1000:
            logger.info(f"[{network}] Dataset too small for GNN ({len(X)} rows), skipping")
            return None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            extractor = GraphFeatureExtractor(
                in_channels=X.shape[1],
                hidden=64,
                window_size=48,
                device=device,
            )
            extractor._pretrain(X.values, y_log.values, epochs=15)
            emb = extractor.extract(X.values)
            cols = [f"gnn_{i}" for i in range(emb.shape[1])]
            gnn_df = pd.DataFrame(emb, columns=cols, index=X.index)
            logger.info(f"[{network}] GNN embeddings added: {emb.shape[1]} dims")
            return gnn_df
        except Exception as e:
            logger.warning(f"[{network}] GNN extraction failed: {e}")
            return None

    def train(self, data: pd.DataFrame, network: str) -> Dict:
        logger.info(f"Training model for {network}")

        block_time = self._detect_block_time(data, network)
        blocks_per_hour = max(1, int(round(3600 / block_time)))

        data_engineered = self.feature_engineer.create_all_features(data.copy())
        X, feature_cols = self._prepare_features(data_engineered)

        valid_mask = (
            data_engineered["gas_price_gwei"].notna()
            & (data_engineered["gas_price_gwei"] > 0)
            & np.isfinite(data_engineered["gas_price_gwei"])
        )

        X = X[valid_mask].reset_index(drop=True)
        data_filtered = data_engineered[valid_mask].reset_index(drop=True)
        y_log = np.log1p(data_filtered["gas_price_gwei"])

        gnn_df = self._try_build_gnn_features(X, y_log, network)
        if gnn_df is not None:
            X = pd.concat([X, gnn_df], axis=1)
            feature_cols = list(X.columns)

        test_size = blocks_per_hour * self.test_hours
        train_size = len(X) - test_size

        if train_size < 100:
            raise ValueError(
                f"[{network}] Not enough data: {len(X)} blocks, "
                f"need at least {test_size + 100}"
            )

        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_log_train = y_log.iloc[:train_size]
        y_log_test = y_log.iloc[train_size:]

        sample_weights = self._calculate_weights(data_filtered, train_size)

        self.log_price_last[network] = {
            "train_end": float(y_log_train.iloc[-1]),
            "test_start": float(y_log_test.iloc[0]),
        }

        logger.info(
            f"[{network}] block_time={block_time:.1f}s blocks_per_hour={blocks_per_hour} "
            f"train={train_size} ({train_size / blocks_per_hour:.1f}h) "
            f"test={test_size} ({self.test_hours}h)"
        )

        horizon_models: Dict[int, Any] = {}
        quantile_horizon_models: Dict[int, Tuple[Any, Any]] = {}
        all_metrics: Dict[int, Dict] = {}

        for h in range(1, self.test_hours + 1):
            zone = "short" if h <= 8 else "medium" if h <= 16 else "long"
            shift = h * blocks_per_hour

            if train_size - shift < 50:
                logger.warning(f"[{network}] h={h}: shift={shift} too large, skipping")
                continue

            y_future_log = y_log.shift(-shift)
            y_return_h = y_future_log - y_log

            y_train_h = y_return_h.iloc[:train_size]
            y_test_h = y_return_h.iloc[train_size:]
            y_log_test_h = y_future_log.iloc[train_size:]

            valid_train = y_train_h.notna()
            valid_test = y_test_h.notna() & y_log_test_h.notna()

            if valid_train.sum() < 50:
                logger.warning(f"[{network}] h={h}: not enough train samples, skipping")
                continue

            logger.info(
                f"[{network}] h={h}/{self.test_hours} (zone={zone}) "
                f"train_samples={valid_train.sum()}"
            )

            params = self._get_params_for_horizon(network, h)
            model = HistGradientBoostingRegressor(**params)
            sw = sample_weights[valid_train.values] if sample_weights is not None else None

            try:
                model.fit(X_train[valid_train], y_train_h[valid_train], sample_weight=sw)
            except Exception as e:
                logger.warning(f"[{network}] h={h}: fit failed — {e}")
                continue

            q10_model, q90_model = None, None
            if zone in ("medium", "long"):
                try:
                    q10_params = self._get_quantile_params(network, h, 0.1)
                    q10_model = HistGradientBoostingRegressor(**q10_params)
                    q10_model.fit(X_train[valid_train], y_train_h[valid_train])

                    q90_params = self._get_quantile_params(network, h, 0.9)
                    q90_model = HistGradientBoostingRegressor(**q90_params)
                    q90_model.fit(X_train[valid_train], y_train_h[valid_train])
                except Exception as e:
                    logger.warning(f"[{network}] h={h}: quantile fit failed — {e}")

            if valid_test.sum() > 0:
                y_log_current_test = y_log_test[valid_test.values]
                pred_returns = model.predict(X_test[valid_test])
                y_pred = np.expm1(y_log_current_test.values + pred_returns)
                y_true = np.expm1(y_log_test_h[valid_test].values)

                mae = float(np.mean(np.abs(y_true - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
                mask = y_true != 0
                mape = (
                    float(
                        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                    )
                    if mask.any()
                    else 0.0
                )
                naive_mae = self._naive_baseline_mae(y_true)
                skill = float(1 - mae / naive_mae) if naive_mae > 0 else 0.0

                all_metrics[h] = {
                    "mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
                    "naive_mae": naive_mae, "skill_score": skill,
                }
                logger.info(
                    f"[{network}] h={h}h MAE={mae:.4f} R²={r2:.4f} "
                    f"MAPE={mape:.2f}% skill={skill:.3f} (naive={naive_mae:.4f})"
                )

            horizon_models[h] = model
            if q10_model is not None and q90_model is not None:
                quantile_horizon_models[h] = (q10_model, q90_model)

        self.models[network] = horizon_models
        self.quantile_models[network] = quantile_horizon_models
        self.feature_columns[network] = feature_cols
        self._save_model(network)

        return all_metrics

    def train_all_networks(self, data: pd.DataFrame) -> Dict:
        metrics = {}
        network_data = data[data["network"] == "ethereum"].copy()
        if len(network_data) >= 100:
            all_h_metrics = self.train(network_data, "ethereum")
            metrics["ethereum"] = all_h_metrics.get(1, {"mae": 0, "rmse": 0, "r2": 0, "mape": 0})
            metrics["ethereum"]["all_horizons"] = all_h_metrics
        else:
            logger.warning(f"Insufficient data for ethereum: {len(network_data)} blocks")
        return metrics

    def _get_current_log_price(self, network: str, features: pd.DataFrame) -> Optional[float]:
        if "gas_log" in features.columns:
            val = features["gas_log"].iloc[-1]
            if np.isfinite(val):
                return float(val)
        if "gas_price_gwei" in features.columns:
            val = features["gas_price_gwei"].iloc[-1]
            if val > 0 and np.isfinite(val):
                return float(np.log1p(val))
        cached = self.log_price_last.get(network, {})
        return cached.get("test_start")

    def predict(self, network: str, features: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        if network not in self.models:
            raise ValueError(f"No model for {network}")

        horizon_models = self.models[network]
        closest_h = min(horizon_models.keys(), key=lambda x: abs(x - horizon))
        model = horizon_models[closest_h]

        feature_cols = self.feature_columns[network]
        X = features.copy()
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        log_price_now = self._get_current_log_price(network, features)
        if log_price_now is None:
            log_price_now = 0.0

        pred_returns = model.predict(X)
        return np.expm1(log_price_now + pred_returns)

    def predict_with_interval(
        self, network: str, features: pd.DataFrame, horizon: int = 1
    ) -> Dict[str, np.ndarray]:
        feature_cols = self.feature_columns[network]
        X = features.copy()
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        log_price_now = self._get_current_log_price(network, features)
        if log_price_now is None:
            log_price_now = 0.0

        closest_h = min(self.models[network].keys(), key=lambda x: abs(x - horizon))
        point_returns = self.models[network][closest_h].predict(X)
        point = np.expm1(log_price_now + point_returns)

        q_models = self.quantile_models.get(network, {}).get(closest_h)
        if q_models is not None:
            q10_model, q90_model = q_models
            lower = np.expm1(log_price_now + q10_model.predict(X))
            upper = np.expm1(log_price_now + q90_model.predict(X))
        else:
            uncertainty = 0.10 + 0.008 * horizon
            lower = point * (1 - uncertainty)
            upper = point * (1 + uncertainty)

        return {"point": point, "lower": lower, "upper": upper}

    def check_drift(self, network: str, recent_data: pd.DataFrame) -> Dict:
        if network not in self.models:
            return {"status": "no_model", "network": network}
        try:
            data_engineered = self.feature_engineer.create_all_features(recent_data.copy())
            X, _ = self._prepare_features(data_engineered)
            valid_mask = (
                data_engineered["gas_price_gwei"].notna()
                & (data_engineered["gas_price_gwei"] > 0)
                & np.isfinite(data_engineered["gas_price_gwei"])
            )
            if valid_mask.sum() < 10:
                return {"status": "insufficient_data", "network": network}

            feature_cols = self.feature_columns[network]
            X_valid = X[valid_mask].copy()
            for col in feature_cols:
                if col not in X_valid.columns:
                    X_valid[col] = 0
            X_valid = X_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

            y_true = data_engineered["gas_price_gwei"][valid_mask].values

            log_price_now = self._get_current_log_price(
                network, data_engineered[valid_mask].reset_index(drop=True)
            )
            if log_price_now is None:
                log_price_now = float(np.log1p(y_true[0])) if len(y_true) > 0 else 0.0

            model = list(self.models[network].values())[0]
            pred_returns = model.predict(X_valid)
            y_pred = np.expm1(log_price_now + pred_returns)

            mae = float(np.mean(np.abs(y_true - y_pred)))
            mape = float(
                np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
            )
            status = "ok" if mape < 30 else "drift_detected"
            return {
                "status": status,
                "network": network,
                "mae": round(mae, 4),
                "mape": round(mape, 2),
            }
        except Exception as exc:
            return {"status": "error", "network": network, "error": str(exc)}
        