from pydantic import BaseModel
from typing import Dict, List, Optional


class DeploymentOption(BaseModel):
    network:                    str
    deploy_in_hours:            int
    predicted_gas_price_gwei:   float
    deployment_cost_usd:        float
    daily_operational_cost_usd: float
    savings_vs_current_pct:     float


class PredictionResult(BaseModel):
    all_options: List[DeploymentOption]
    best_option: DeploymentOption
    timestamp:   str


class HealthResponse(BaseModel):
    status:          str
    ml_model_loaded: bool
    timestamp:       str
    