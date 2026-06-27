"""
api/routers/predictions.py
Prediction data endpoints:
  GET /predictions              — recent predictions for a pair (pair + limit)
  GET /predictions/latest       — most recent prediction
  GET /predictions/accuracy     — coverage accuracy stats
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import csv
import io
from datetime import datetime

from api import storage
from api.config import PAIR
from api.schemas import PredictionOut, AccuracyOut, RecentPredictionsOut

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/accuracy", response_model=AccuracyOut, summary="Coverage accuracy stats")
def get_accuracy():
    return storage.get_accuracy_stats()


@router.get("/latest", response_model=PredictionOut, summary="Latest prediction")
def get_latest():
    row = storage.get_latest_prediction()
    if row is None:
        raise HTTPException(status_code=404, detail="No predictions yet.")
    return _to_schema(row)


@router.get("", response_model=RecentPredictionsOut, summary="Recent predictions for a pair")
def get_predictions(
    pair: str = Query(PAIR, description="Currency pair, e.g. EURUSD"),
    limit: int = Query(100, ge=1, le=1000, description="Number of recent predictions"),
):
    rows = storage.get_recent_predictions(pair, limit)
    return RecentPredictionsOut(
        pair=pair,
        count=len(rows),
        data=[_to_schema(r) for r in rows],
    )


def _parse_utc_query_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid UTC date/datetime format for query param: '{value}'. "
                "Use YYYY-MM-DD or YYYY-MM-DDThh:mm:ss"
            ),
        )


@router.get("/export.csv", summary="Export predictions with candles as CSV")
def export_predictions_csv(
    pair: str = Query(PAIR, description="Currency pair, e.g. EURUSD"),
    start: str | None = Query(None, description="UTC start date/datetime, inclusive, e.g. 2026-05-26 or 2026-06-15 12:30:00"),
    end: str | None = Query(None, description="UTC end date/datetime, exclusive, e.g. 2026-06-16 or 2026-06-15 12:30:00"),
):
    start_dt = _parse_utc_query_datetime(start)
    end_dt = _parse_utc_query_datetime(end)

    rows = storage.iter_predictions_with_candles(pair=pair, start=start_dt, end=end_dt)

    headers = [
        "time", "target_time", "actual_upper", "actual_lower",
        "actual_upper_close", "actual_lower_close",
        "upper_q10_pip", "upper_q50_pip", "upper_q90_pip",
        "lower_q10_pip", "lower_q50_pip", "lower_q90_pip",
        "current_open", "current_high", "current_low", "current_close",
        "upper_q10_price", "lower_q10_price",
        "upper_q50_price", "lower_q50_price",
        "upper_q90_price", "lower_q90_price",
        "next_candle_open", "next_candle_high", "next_candle_low", "next_candle_close",
    ]

    def generate():
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=headers)
        writer.writeheader()
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)

    filename = f"predictions_{pair}.csv"
    if start_dt or end_dt:
        range_label = f"_{start or 'start'}_{end or 'end'}"
        filename = f"predictions_{pair}{range_label}.csv"

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _to_schema(row: dict) -> PredictionOut:
    """Convert a raw storage dict to the PredictionOut schema."""
    return PredictionOut(
        id=int(row.get("id", 0)),
        pair=row.get("pair"),
        timeframe=row.get("timeframe"),
        current_candle_time=row.get("current_candle_time"),
        prediction_start_time=row.get("prediction_start_time"),
        prediction_expiry_time=row.get("prediction_expiry_time"),
        target_time=row.get("target_time"),
        upper_q10_pip=row.get("upper_q10_pip"),
        upper_q50_pip=row.get("upper_q50_pip"),
        upper_q90_pip=row.get("upper_q90_pip"),
        lower_q10_pip=row.get("lower_q10_pip"),
        lower_q50_pip=row.get("lower_q50_pip"),
        lower_q90_pip=row.get("lower_q90_pip"),
        upper_q10_price=row.get("upper_q10_price"),
        upper_q50_price=row.get("upper_q50_price"),
        upper_q90_price=row.get("upper_q90_price"),
        lower_q10_price=row.get("lower_q10_price"),
        lower_q50_price=row.get("lower_q50_price"),
        lower_q90_price=row.get("lower_q90_price"),
        actual_upper=row.get("actual_upper"),
        actual_lower=row.get("actual_lower"),
        live_close=row.get("live_close"),
    )
