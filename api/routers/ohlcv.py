"""
api/routers/ohlcv.py
GET /ohlcv — recent candles for a pair (pair + limit only).
"""
from fastapi import APIRouter, Query

from api import storage
from api.config import PAIR
from api.schemas import OhlcvListOut, OhlcvOut

router = APIRouter(prefix="/ohlcv", tags=["OHLCV"])


@router.get("", response_model=OhlcvListOut, summary="Recent OHLCV candles for a pair")
def get_ohlcv(
    pair: str = Query(PAIR, description="Currency pair, e.g. EURUSD"),
    limit: int = Query(100, ge=1, le=1000, description="Number of recent candles"),
):
    rows = storage.get_ohlcv(pair, limit)
    return OhlcvListOut(
        pair=pair,
        count=len(rows),
        data=[OhlcvOut(**r) for r in rows],
    )
