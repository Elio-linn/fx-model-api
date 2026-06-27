"""
api/routers/status.py
GET /status — bot running state and summary accuracy stats.
"""
from datetime import datetime, timezone
from fastapi import APIRouter

from api import state, storage
from api.config import MODEL_VERSION
from api.schemas import BotStatusOut
from api.scheduler import get_next_run_seconds

router = APIRouter(prefix="/status", tags=["Status"])


@router.get("", response_model=BotStatusOut, summary="Bot status and accuracy summary")
def get_status():
    running = state.is_running()
    last_run = state.get_last_run()
    last_candle_time, _ = state.get_last_candle()
    next_sec = get_next_run_seconds()
    accuracy = storage.get_accuracy_stats()
    status_code, status_msg, market_open = state.get_status()
    model_status = state.get_model_status()

    last_run_str = last_run.isoformat() if last_run else None

    if not running:
        message = "Bot is stopped"
    elif status_msg:
        message = status_msg
    else:
        message = "Bot is running"

    return BotStatusOut(
        running=running,
        last_run_utc=last_run_str,
        last_candle_time=last_candle_time,
        next_run_in_seconds=next_sec,
        total_predictions=accuracy["total_evaluated"] if accuracy else 0,
        upper_coverage_pct=accuracy.get("upper_coverage_pct"),
        lower_coverage_pct=accuracy.get("lower_coverage_pct"),
        market_open=market_open,
        status=status_code,
        model_status=model_status,
        model_version=MODEL_VERSION,
        message=message,
    )
