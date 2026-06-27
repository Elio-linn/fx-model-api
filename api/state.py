"""
api/state.py
Shared mutable bot state across the FastAPI app and background scheduler.
Uses a simple dict protected by threading.Lock.
"""
import threading
from datetime import datetime
from typing import Optional

_lock = threading.Lock()

_state = {
    "running": False,
    "last_run_utc": None,           # datetime | None
    "last_candle_time": None,       # str | None
    "last_prediction_id": None,     # int | None — pending actuals update
    "last_candle_row": None,        # pd.Series | None — raw OHLCV of last candle
    "error_count": 0,
    "market_open": True,            # bool — FX market hours
    "last_status": "ok",            # "ok" | "market_closed" | "candle_missing" | "error"
    "last_status_message": "",      # str — human-readable detail
    "model_status": "inactive",     # "active" | "inactive" — data API reachability
}


def set_running(value: bool):
    with _lock:
        _state["running"] = value


def is_running() -> bool:
    with _lock:
        return _state["running"]


def set_last_run(dt: datetime):
    with _lock:
        _state["last_run_utc"] = dt


def get_last_run() -> Optional[datetime]:
    with _lock:
        return _state["last_run_utc"]


def set_last_candle(time_str: str, candle_row):
    with _lock:
        _state["last_candle_time"] = time_str
        _state["last_candle_row"] = candle_row


def get_last_candle():
    with _lock:
        return _state["last_candle_time"], _state["last_candle_row"]


def set_last_prediction_id(pid: Optional[int]):
    with _lock:
        _state["last_prediction_id"] = pid


def get_last_prediction_id() -> Optional[int]:
    with _lock:
        return _state["last_prediction_id"]


def increment_error():
    with _lock:
        _state["error_count"] += 1


def get_error_count() -> int:
    with _lock:
        return _state["error_count"]


def set_status(code: str, message: str = "", market_open: Optional[bool] = None):
    with _lock:
        _state["last_status"] = code
        _state["last_status_message"] = message
        if market_open is not None:
            _state["market_open"] = market_open


def get_status():
    """Return (code, message, market_open)."""
    with _lock:
        return (
            _state["last_status"],
            _state["last_status_message"],
            _state["market_open"],
        )


def set_model_status(value: str):
    with _lock:
        _state["model_status"] = value


def get_model_status() -> str:
    with _lock:
        return _state["model_status"]
