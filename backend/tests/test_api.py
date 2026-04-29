import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


def make_gas_df(n=300):
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "gas_price_gwei": [round(20.0 + i * 0.05, 2) for i in range(n)],
    })


def make_boxplot_df():
    dates = pd.date_range("2024-01-01", periods=300, freq="1h")
    return pd.DataFrame({
        "datetime": dates,
        "gas_price_gwei": [25.0 + (i % 10) for i in range(300)],
    })


def make_mock_network_client(gas_used=15_000_000, gas_limit=30_000_000, tx=150):
    mock = AsyncMock()
    mock.get_gas_oracle.return_value = {
        "safe_gas_price": 10,
        "propose_gas_price": 15,
        "fast_gas_price": 20,
    }
    mock.get_latest_block_number.return_value = 19_000_000
    mock.get_block_details.return_value = {
        "gas_used": gas_used,
        "gas_limit": gas_limit,
        "transaction_count": tx,
    }
    mock.get_pending_count.return_value = 5000
    return mock


class TestHealth:
    def test_status_200(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_returns_operational(self):
        assert client.get("/api/health").json()["status"] == "operational"

    def test_ml_model_loaded_is_bool(self):
        assert isinstance(client.get("/api/health").json()["ml_model_loaded"], bool)

    def test_timestamp_is_string(self):
        assert isinstance(client.get("/api/health").json()["timestamp"], str)

    def test_no_extra_error_fields(self):
        data = client.get("/api/health").json()
        assert "detail" not in data


class TestDataCollection:
    def test_returns_200(self):
        assert client.post("/api/collect-data").status_code == 200

    def test_status_is_collection_initiated(self):
        assert client.post("/api/collect-data").json()["status"] == "collection_initiated"

    def test_has_timestamp(self):
        assert "timestamp" in client.post("/api/collect-data").json()

    def test_timestamp_is_string(self):
        assert isinstance(client.post("/api/collect-data").json()["timestamp"], str)


class TestGasSignalMocked:
    def test_returns_200(self):
        df = make_gas_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert client.get("/api/gas-signal").status_code == 200

    def test_current_gwei_is_float(self):
        df = make_gas_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert isinstance(client.get("/api/gas-signal").json()["current_gwei"], float)

    def test_median_24h_positive(self):
        df = make_gas_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert client.get("/api/gas-signal").json()["median_24h"] > 0

    def test_zscore_is_numeric(self):
        df = make_gas_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert isinstance(client.get("/api/gas-signal").json()["zscore"], (int, float))

    def test_all_fields_present(self):
        df = make_gas_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            data = client.get("/api/gas-signal").json()
            for field in ("current_gwei", "median_24h", "zscore", "pct_vs_median"):
                assert field in data

    def test_no_data_returns_404(self):
        with patch("api.main.Path.glob", return_value=[]):
            assert client.get("/api/gas-signal").status_code == 404


class TestHeatmapMocked:
    def test_returns_200(self):
        df = make_gas_df(500)
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert client.get("/api/heatmap").status_code == 200

    def test_matrix_is_7_rows(self):
        df = make_gas_df(500)
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert len(client.get("/api/heatmap").json()["matrix"]) == 7

    def test_each_row_has_24_columns(self):
        df = make_gas_df(500)
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            matrix = client.get("/api/heatmap").json()["matrix"]
            assert all(len(row) == 24 for row in matrix)

    def test_hours_is_0_to_23(self):
        df = make_gas_df(500)
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert client.get("/api/heatmap").json()["hours"] == list(range(24))

    def test_days_has_7_elements(self):
        df = make_gas_df(500)
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert len(client.get("/api/heatmap").json()["days"]) == 7

    def test_no_data_returns_404(self):
        with patch("api.main.Path.glob", return_value=[]):
            assert client.get("/api/heatmap").status_code == 404


class TestBoxplotMocked:
    def test_returns_200(self):
        df = make_boxplot_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert client.get("/api/boxplot").status_code == 200

    def test_data_is_list(self):
        df = make_boxplot_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            assert isinstance(client.get("/api/boxplot").json()["data"], list)

    def test_item_has_all_fields(self):
        df = make_boxplot_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            items = client.get("/api/boxplot").json()["data"]
            assert len(items) > 0
            for field in ("label", "min", "q1", "median", "q3", "max", "mean"):
                assert field in items[0]

    def test_q1_lte_median_lte_q3(self):
        df = make_boxplot_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            for item in client.get("/api/boxplot").json()["data"]:
                assert item["q1"] <= item["median"] <= item["q3"]

    def test_min_lte_max(self):
        df = make_boxplot_df()
        with patch("api.main.pd.read_csv", return_value=df), \
             patch("api.main.Path.glob", return_value=[Path("fake.csv")]):
            for item in client.get("/api/boxplot").json()["data"]:
                assert item["min"] <= item["max"]

    def test_no_data_returns_404(self):
        with patch("api.main.Path.glob", return_value=[]):
            assert client.get("/api/boxplot").status_code == 404


class TestNetworkStatsMocked:
    def test_returns_200(self):
        with patch("api.main.predictor.client", make_mock_network_client()):
            assert client.get("/api/network-stats").status_code == 200

    def test_all_gas_fields_present(self):
        with patch("api.main.predictor.client", make_mock_network_client()):
            data = client.get("/api/network-stats").json()
            for field in ("safe_gas", "propose_gas", "fast_gas", "last_block"):
                assert field in data

    def test_utilization_50_percent(self):
        with patch("api.main.predictor.client", make_mock_network_client(
            gas_used=15_000_000, gas_limit=30_000_000
        )):
            assert client.get("/api/network-stats").json()["avg_utilization"] == 50.0

    def test_last_block_is_int(self):
        with patch("api.main.predictor.client", make_mock_network_client()):
            assert isinstance(client.get("/api/network-stats").json()["last_block"], int)

    def test_fast_gas_gte_safe_gas(self):
        with patch("api.main.predictor.client", make_mock_network_client()):
            data = client.get("/api/network-stats").json()
            assert data["fast_gas"] >= data["safe_gas"]

    def test_pending_queue_present(self):
        with patch("api.main.predictor.client", make_mock_network_client()):
            assert "pending_queue" in client.get("/api/network-stats").json()
            