import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self):
        pass

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'network' not in df.columns or df.empty:
            return df

        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        df['gas_log'] = np.log1p(df['gas_price_gwei'])
        df['network_id'] = 0

        df = self._add_time_features(df)
        df = self._add_eip1559_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_volatility_features(df)
        df = self._add_har_features(df)
        df = self._add_momentum_features(df)
        df = self._add_microstructure_features(df)
        df = self._add_trend_features(df)
        df = self._add_percentile_features(df)
        df = self._add_regime_features(df)
        df = self._add_interaction_features(df)

        df = df.ffill().fillna(0)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_us_peak'] = ((df['hour'] >= 14) & (df['hour'] <= 22)).astype(int)
        df['is_asia_peak'] = ((df['hour'] >= 1) & (df['hour'] <= 9)).astype(int)
        df['is_low_traffic'] = ((df['hour'] >= 2) & (df['hour'] <= 7)).astype(int)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
        df['month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 30)

        hours_to_next_low = (2 - df['hour']).clip(lower=0)
        hours_to_next_low[df['hour'] >= 7] = 24 - df['hour'][df['hour'] >= 7] + 2
        df['hours_to_low_traffic'] = hours_to_next_low
        return df

    def _add_eip1559_features(self, df: pd.DataFrame) -> pd.DataFrame:
        eip_delta = 0.125 * (2 * df['utilization'] - 1)
        df['eip1559_delta'] = eip_delta
        df['util_vs_target'] = df['utilization'] - 0.5
        df['eip1559_pressure'] = df['gas_price_gwei'] * eip_delta

        log_delta_clipped = np.log1p(eip_delta.clip(-0.125, 0.125))
        df['eip1559_next_log'] = df['gas_log'] + log_delta_clipped

        for n_blocks, label in [(150, '1h'), (450, '3h'), (900, '6h'), (1800, '12h')]:
            proj = (df['gas_log'] + n_blocks * log_delta_clipped).clip(-5, 15)
            df[f'eip1559_proj_{label}'] = proj

        util_safe = df['utilization'].replace(0, 1e-9)
        df['eip1559_equilibrium_log'] = df['gas_log'] - np.log1p(
            0.125 * (2 * util_safe - 1) * 150
        ).clip(-5, 5)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('datetime')
        df = df[~df.index.duplicated(keep='last')]

        lag_specs = {
            '1h':  pd.Timedelta('30min'),
            '3h':  pd.Timedelta('1h'),
            '6h':  pd.Timedelta('2h'),
            '12h': pd.Timedelta('3h'),
            '24h': pd.Timedelta('6h'),
        }

        for name, freq in lag_specs.items():
            shifted_idx = df.index - pd.Timedelta(name)
            gas_shifted = df['gas_price_gwei'].reindex(shifted_idx, method='nearest', tolerance=freq)
            util_shifted = df['utilization'].reindex(shifted_idx, method='nearest', tolerance=freq)
            log_shifted = df['gas_log'].reindex(shifted_idx, method='nearest', tolerance=freq)

            gas_shifted.index = df.index
            util_shifted.index = df.index
            log_shifted.index = df.index

            df[f'gas_lag_{name}'] = gas_shifted.values
            df[f'util_lag_{name}'] = util_shifted.values
            df[f'log_return_{name}'] = (df['gas_log'].values - log_shifted.values)

        df = df.reset_index()
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('datetime')
        df = df[~df.index.duplicated(keep='last')]

        for w in ['1h', '3h', '6h', '12h', '24h']:
            df[f'ma_{w}'] = df['gas_price_gwei'].rolling(w, min_periods=1).mean()
            df[f'util_ma_{w}'] = df['utilization'].rolling(w, min_periods=1).mean()
            df[f'log_ma_{w}'] = df['gas_log'].rolling(w, min_periods=1).mean()

        if 'tx_count' in df.columns:
            df['tx_count_ma_1h'] = df['tx_count'].rolling('1h', min_periods=1).mean()
            df['tx_count_ma_6h'] = df['tx_count'].rolling('6h', min_periods=1).mean()

        df = df.reset_index()
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('datetime')
        df = df[~df.index.duplicated(keep='last')]

        for w in ['1h', '6h', '12h', '24h']:
            roll = df['gas_price_gwei'].rolling(w, min_periods=2)
            df[f'volatility_{w}'] = roll.std()
            df[f'iqr_{w}'] = roll.quantile(0.75) - roll.quantile(0.25)
            df[f'median_{w}'] = roll.median()

        df = df.reset_index()
        return df

    def _add_har_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('datetime')
        df = df[~df.index.duplicated(keep='last')]

        log_ret = df['gas_log'].diff()
        sq_ret = log_ret ** 2

        df['rv_1h'] = sq_ret.rolling('1h', min_periods=3).sum()
        df['rv_6h'] = sq_ret.rolling('6h', min_periods=10).mean()
        df['rv_12h'] = sq_ret.rolling('12h', min_periods=20).mean()
        df['rv_24h'] = sq_ret.rolling('24h', min_periods=40).mean()
        df['rv_7d'] = sq_ret.rolling('168h', min_periods=200).mean()

        df['log_rv_1h'] = np.log1p(df['rv_1h'])
        df['log_rv_24h'] = np.log1p(df['rv_24h'])
        df['log_rv_7d'] = np.log1p(df['rv_7d'])

        df['rv_ratio_1h_24h'] = (df['rv_1h'] / (df['rv_24h'] + 1e-12)).clip(0, 100)
        df['rv_ratio_6h_7d'] = (df['rv_6h'] / (df['rv_7d'] + 1e-12)).clip(0, 100)

        util_ret = df['utilization'].diff()
        df['rv_util_1h'] = (util_ret ** 2).rolling('1h', min_periods=3).sum()
        df['rv_util_6h'] = (util_ret ** 2).rolling('6h', min_periods=10).mean()

        df = df.reset_index()
        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('datetime')
        df = df[~df.index.duplicated(keep='last')]

        for w in ['1h', '3h', '6h', '12h', '24h']:
            roll_mean = df['gas_log'].rolling(w, min_periods=1).mean()
            df[f'momentum_{w}'] = df['gas_log'] - roll_mean

        for w in ['6h', '12h', '24h']:
            roll_mean = df['gas_log'].rolling(w, min_periods=5).mean()
            roll_std = df['gas_log'].rolling(w, min_periods=5).std().replace(0, 1e-9)
            df[f'zscore_{w}'] = ((df['gas_log'] - roll_mean) / roll_std).clip(-5, 5)

        lag1_ret = df['gas_log'].diff(1)
        abs_sum = lag1_ret.abs().rolling('1h', min_periods=5).sum().replace(0, 1e-9)
        df['autocorr_sign_1h'] = lag1_ret.rolling('1h', min_periods=5).sum() / abs_sum

        df = df.reset_index()
        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('datetime')
        df = df[~df.index.duplicated(keep='last')]

        priority_min_candidates = ['min_priority_fee', 'min_gas_price', 'priority_fee_min']
        priority_avg_candidates = ['avg_priority_fee', 'avg_gas_price', 'priority_fee_avg']
        priority_max_candidates = ['max_priority_fee', 'max_gas_price', 'priority_fee_max']

        p_min = next((df[c] for c in priority_min_candidates if c in df.columns), None)
        p_avg = next((df[c] for c in priority_avg_candidates if c in df.columns), None)
        p_max = next((df[c] for c in priority_max_candidates if c in df.columns), None)

        if p_min is not None and p_max is not None:
            df['priority_spread'] = (p_max - p_min).clip(0, 500)
            if p_avg is not None:
                df['priority_skew'] = ((p_avg - p_min) / (df['priority_spread'] + 1e-9)).clip(0, 1)
                base_gwei = df['gas_price_gwei'] + 1e-9
                df['priority_to_base_ratio'] = (p_avg / base_gwei).clip(0, 100)
        elif p_avg is not None:
            df['priority_spread'] = p_avg.clip(0, 500)
            df['priority_skew'] = 0.5

        gas_change = df['gas_log'].diff().abs()
        util_change = df['utilization'].diff().abs().replace(0, np.nan)
        df['kyle_lambda'] = (gas_change / util_change).clip(0, 1000).fillna(0)
        df['kyle_lambda_1h'] = df['kyle_lambda'].rolling('1h', min_periods=3).mean()
        df['kyle_lambda_6h'] = df['kyle_lambda'].rolling('6h', min_periods=10).mean()

        if 'tx_count' in df.columns:
            safe_tx = df['tx_count'].replace(0, np.nan)
            df['amihud'] = (gas_change / safe_tx).clip(0, 10).fillna(0)
            df['amihud_1h'] = df['amihud'].rolling('1h', min_periods=3).mean()

        df = df.reset_index()
        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['trend_1h_vs_24h'] = df['ma_1h'] - df['ma_24h']
        df['trend_3h_vs_24h'] = df['ma_3h'] - df['ma_24h']
        df['trend_6h_vs_24h'] = df['ma_6h'] - df['ma_24h']

        lr1 = df.get('log_return_1h', pd.Series(0.0, index=df.index))
        lr6 = df.get('log_return_6h', pd.Series(0.0, index=df.index))
        df['gas_log_change_1h'] = lr1
        df['gas_log_change_6h'] = lr6
        df['util_change_1h'] = df['utilization'] - df.get('util_lag_1h', df['utilization'])
        return df

    def _add_percentile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('datetime')
        df = df[~df.index.duplicated(keep='last')]

        for w in ['6h', '24h', '168h']:
            roll = df['gas_price_gwei'].rolling(w, min_periods=1)
            rmin = roll.min()
            rmax = roll.max()
            df[f'gas_percentile_{w}'] = ((df['gas_price_gwei'] - rmin) / (rmax - rmin + 1e-9)).clip(0, 1)

        df = df.reset_index()
        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('datetime')
        df = df[~df.index.duplicated(keep='last')]

        median_24h = df['gas_price_gwei'].rolling('24h', min_periods=1).median()
        df['is_spike'] = (df['gas_price_gwei'] > median_24h * 2.0).astype(int)
        df['spike_memory_6h'] = df['is_spike'].rolling('6h', min_periods=1).max()
        df['spike_memory_12h'] = df['is_spike'].rolling('12h', min_periods=1).max()
        df['spike_intensity'] = (df['gas_price_gwei'] / (median_24h + 1e-9)).clip(0, 10)

        log_median_24h = df['gas_log'].rolling('24h', min_periods=1).median()
        df['log_reversion_signal'] = df['gas_log'] - log_median_24h

        df['util_above_target'] = (df['utilization'] > 0.5).astype(int)
        df['util_streak'] = df['util_above_target'].rolling('1h', min_periods=1).sum()
        df['util_streak_3h'] = df['util_above_target'].rolling('3h', min_periods=1).mean()

        util_ma_1h = df['utilization'].rolling('1h', min_periods=1).mean()
        util_ma_3h = df['utilization'].rolling('3h', min_periods=1).mean()
        df['util_acceleration'] = util_ma_1h - util_ma_3h

        util_std_3h = df['utilization'].rolling('3h', min_periods=2).std().replace(0, np.nan)
        gas_std_3h = df['gas_price_gwei'].rolling('3h', min_periods=2).std()
        df['gas_elasticity_3h'] = (gas_std_3h / util_std_3h).clip(0, 1000)

        df = df.reset_index()
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour_sin_x_log'] = df['hour_sin'] * df['gas_log']
        df['hour_cos_x_log'] = df['hour_cos'] * df['gas_log']
        df['weekend_x_log'] = df['is_weekend'] * df['gas_log']
        df['us_peak_x_log'] = df['is_us_peak'] * df['gas_log']
        df['low_traffic_x_log'] = df['is_low_traffic'] * df['gas_log']

        df['zscore_24h_x_hour_sin'] = df.get('zscore_24h', 0) * df['hour_sin']
        df['zscore_24h_x_weekend'] = df.get('zscore_24h', 0) * df['is_weekend']

        df['rv_1h_x_hour_sin'] = df.get('rv_1h', 0) * df['hour_sin']
        df['spike_x_us_peak'] = df.get('is_spike', 0) * df['is_us_peak']
        df['spike_x_hour_sin'] = df.get('is_spike', 0) * df['hour_sin']

        df['eip1559_proj_1h_x_util'] = df.get('eip1559_proj_1h', df['gas_log']) * df['utilization']
        df['reversion_x_hour_cos'] = df.get('log_reversion_signal', 0) * df['hour_cos']

        return df
    