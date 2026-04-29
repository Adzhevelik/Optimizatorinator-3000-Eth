import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from api.schemas import HealthResponse
from ml.predictor import GasPricePredictor
from ml.data_collector import DataCollector
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = GasPricePredictor()
collector = DataCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.initialize()
    yield


app = FastAPI(
    title="Ethereum Gas Price Optimizer",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="operational",
        ml_model_loaded=predictor.ready,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/network-stats")
async def get_network_stats():
    try:
        gas_oracle   = await predictor.client.get_gas_oracle("ethereum")
        latest_block = await predictor.client.get_latest_block_number("ethereum")

        utilization = 0.0
        tx_count    = 0
        try:
            block = await predictor.client.get_block_details(latest_block)
            if block and block.get("gas_limit", 0) > 0:
                utilization = round(block["gas_used"] / block["gas_limit"] * 100, 1)
                tx_count    = block.get("transaction_count", 0)
        except Exception:
            pass

        pending = 0
        try:
            pending = await predictor.client.get_pending_count()
        except Exception:
            pass

        return {
            "last_block":      latest_block,
            "pending_queue":   pending,
            "avg_block_size":  tx_count,
            "avg_utilization": utilization,
            "safe_gas":        gas_oracle.get("safe_gas_price",    0),
            "propose_gas":     gas_oracle.get("propose_gas_price", 0),
            "fast_gas":        gas_oracle.get("fast_gas_price",    0),
            "timestamp":       datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gas-signal")
async def get_gas_signal():
    data_path = Path(__file__).parent.parent.parent / "data"
    files = sorted(data_path.glob("ethereum_historical_*.csv"), reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No historical data")

    try:
        df = pd.read_csv(files[0], usecols=["datetime", "gas_price_gwei"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        window     = df.tail(7200)
        median_24h = float(window["gas_price_gwei"].median())
        std_24h    = float(window["gas_price_gwei"].std())
        current    = float(df["gas_price_gwei"].iloc[-1])

        zscore = round((current - median_24h) / std_24h, 2) if std_24h > 0 else 0.0
        pct    = round((current - median_24h) / median_24h * 100, 1) if median_24h > 0 else 0.0

        return {
            "current_gwei":  round(current,    2),
            "median_24h":    round(median_24h, 2),
            "zscore":        zscore,
            "pct_vs_median": pct,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drift")
async def check_drift():
    try:
        return await predictor.check_model_drift("ethereum")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/heatmap")
async def get_heatmap():
    data_path = Path(__file__).parent.parent.parent / "data"
    files = sorted(data_path.glob("ethereum_historical_*.csv"), reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No historical data available")

    dfs = []
    for f in files[:2]:
        try:
            dfs.append(pd.read_csv(f, usecols=["datetime", "gas_price_gwei"]))
        except Exception:
            pass

    if not dfs:
        raise HTTPException(status_code=500, detail="Failed to read historical data")

    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"]     = df["datetime"].dt.hour
    df["dow"]      = df["datetime"].dt.dayofweek

    pivot = (
        df.groupby(["dow", "hour"])["gas_price_gwei"]
        .mean()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )

    return {
        "matrix": [[round(v, 2) for v in row] for row in pivot.values.tolist()],
        "days":   ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"],
        "hours":  list(range(24)),
    }


@app.get("/api/boxplot")
async def get_boxplot():
    data_path = Path(__file__).parent.parent.parent / "data"
    files = sorted(data_path.glob("ethereum_historical_*.csv"), reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No historical data available")

    dfs = []
    for f in files[:2]:
        try:
            dfs.append(pd.read_csv(f, usecols=["datetime", "gas_price_gwei"]))
        except Exception:
            pass

    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"]     = df["datetime"].dt.date

    result = []
    for date, group in sorted(df.groupby("date")):
        vals = group["gas_price_gwei"].dropna()
        if len(vals) < 10:
            continue
        q1  = float(vals.quantile(0.25))
        med = float(vals.median())
        q3  = float(vals.quantile(0.75))
        iqr = q3 - q1
        result.append({
            "label":  str(date)[5:],
            "min":    round(max(float(vals.min()), q1  - 1.5 * iqr), 2),
            "q1":     round(q1,  2),
            "median": round(med, 2),
            "q3":     round(q3,  2),
            "max":    round(min(float(vals.max()), q3  + 1.5 * iqr), 2),
            "mean":   round(float(vals.mean()), 2),
        })

    return {"data": result}


@app.get("/api/predictions")
async def get_predictions():
    try:
        predictions = await predictor.predict_next_hours()
        return {
            "network":     "ethereum",
            "predictions": predictions,
            "timestamp":   datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimal-deployment")
async def get_optimal_deployment(gas_estimate: int, tx_per_day: int = 100):
    try:
        result = await predictor.predict_optimal_deployment(
            gas_estimate=gas_estimate,
            tx_per_day=tx_per_day,
            horizons=list(range(1, 25)),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collect-data")
async def trigger_data_collection(background_tasks: BackgroundTasks):
    async def collect():
        try:
            await collector.update_incremental()
        except Exception as e:
            logger.error(f"Data collection failed: {e}")

    background_tasks.add_task(collect)
    return {"status": "collection_initiated", "timestamp": datetime.now().isoformat()}


@app.get("/api/best-minute")
async def get_best_minute(best_hour: int, timezone: str = "UTC"):
    data_path = Path(__file__).parent.parent.parent / "data"
    files = sorted(data_path.glob("ethereum_historical_*.csv"), reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No historical data")

    try:
        import pytz
        from datetime import datetime, timedelta

        try:
            tz = pytz.timezone(timezone)
        except pytz.UnknownTimeZoneError:
            tz = pytz.utc

        gas_oracle   = await predictor.client.get_gas_oracle("ethereum")
        current_gwei = float(gas_oracle.get("propose_gas_price", 0))

        now         = datetime.now(tz)
        target_dt   = now + timedelta(hours=best_hour)
        target_hour = target_dt.hour

        dfs = []
        for f in files[:2]:
            try:
                dfs.append(pd.read_csv(f, usecols=["datetime", "gas_price_gwei"]))
            except Exception:
                pass

        df             = pd.concat(dfs, ignore_index=True)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(tz)
        df["hour"]     = df["datetime"].dt.hour
        df["minute"]   = df["datetime"].dt.minute

        hour_df = df[df["hour"] == target_hour]
        if len(hour_df) < 10:
            raise HTTPException(status_code=404, detail="Not enough data for this hour")

        by_minute = (
            hour_df.groupby("minute")["gas_price_gwei"]
            .mean()
            .reset_index()
            .sort_values("gas_price_gwei")
        )

        best_minute = int(by_minute.iloc[0]["minute"])
        best_gwei   = round(float(by_minute.iloc[0]["gas_price_gwei"]), 2)
        worst_gwei  = round(float(by_minute["gas_price_gwei"].max()), 2)
        savings_pct = round((worst_gwei - best_gwei) / worst_gwei * 100, 1) if worst_gwei > 0 else 0

        minute_data = [
            {
                "minute":   int(row["minute"]),
                "avg_gwei": round(float(row["gas_price_gwei"]), 2),
            }
            for _, row in by_minute.sort_values("minute").iterrows()
        ]

        return {
            "target_hour":          target_hour,
            "best_hour_offset":     best_hour,
            "best_minute":          best_minute,
            "best_time":            f"{target_hour:02d}:{best_minute:02d}",
            "avg_gwei":             best_gwei,
            "savings_vs_worst_pct": savings_pct,
            "minute_data":          minute_data,
            "timezone":             str(tz),
            "current_time":         now.strftime("%H:%M %Z"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
    