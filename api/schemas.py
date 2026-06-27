"""
api/schemas.py
Pydantic schemas for request/response validation.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class PredictionOut(BaseModel):
    """Single prediction record returned by the API."""
    id: int
    pair: Optional[str] = None
    timeframe: Optional[str] = None
    current_candle_time: Optional[datetime] = None
    prediction_start_time: Optional[datetime] = None
    prediction_expiry_time: Optional[datetime] = None
    target_time: Optional[datetime] = None
    # Predicted range (pips)
    upper_q10_pip: Optional[float] = None
    upper_q50_pip: Optional[float] = None
    upper_q90_pip: Optional[float] = None
    lower_q10_pip: Optional[float] = None
    lower_q50_pip: Optional[float] = None
    lower_q90_pip: Optional[float] = None
    # Predicted range (price levels)
    upper_q10_price: Optional[float] = None
    upper_q50_price: Optional[float] = None
    upper_q90_price: Optional[float] = None
    lower_q10_price: Optional[float] = None
    lower_q50_price: Optional[float] = None
    lower_q90_price: Optional[float] = None
    # Actuals
    actual_upper_open: Optional[float] = None
    actual_lower_open: Optional[float] = None
    actual_upper_close: Optional[float] = None
    actual_lower_close: Optional[float] = None
    live_close: Optional[float] = None


class RecentPredictionsOut(BaseModel):
    pair: str
    count: int
    data: List[PredictionOut]


class OhlcvOut(BaseModel):
    """Single OHLCV candle."""
    pair: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OhlcvListOut(BaseModel):
    pair: str
    count: int
    data: List[OhlcvOut]


class AccuracyOut(BaseModel):
    total_evaluated: int
    upper_coverage_pct: float
    lower_coverage_pct: float
    both_covered_pct: float


class BotStatusOut(BaseModel):
    running: bool
    last_run_utc: Optional[str] = None
    last_candle_time: Optional[str] = None
    next_run_in_seconds: Optional[float] = None
    total_predictions: int
    upper_coverage_pct: Optional[float] = None
    lower_coverage_pct: Optional[float] = None
    market_open: bool = True
    status: str = "ok"              # ok | market_closed | candle_missing | error
    model_status: str = "inactive"  # active (data API reachable) | inactive
    model_version: str = ""         # currently loaded model (from MODEL_VERSION env)
    message: str
