"""
api/storage.py
MongoDB-backed storage for predictions and the three live data sources.

Every source is stored under its own natural key and written with upsert,
so re-fetching the same data across 5-minute cycles deduplicates itself:
  - ohlcv      keyed by (pair, time)
  - gdelt      keyed by (batch_time, currency)   — 15-min batches
  - ff_events  keyed by (event_time, country, title) — irregular events
  - predictions keyed by (pair, target_time), plus an auto-increment id
"""
import logging
import pandas as pd
from datetime import datetime, timezone, timedelta

from pymongo import UpdateOne

from api import db

logger = logging.getLogger(__name__)


def _clean(doc: dict) -> dict:
    """Drop Mongo's _id for JSON serialization."""
    doc.pop("_id", None)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────────────────────────────────────
def append_prediction(record: dict) -> int:
    """
    Insert a new prediction (upsert by pair+target_time, idempotent).
    Returns the assigned auto-increment id.
    """
    target_time = record.get("target_time")
    pair = record.get("pair")

    # Reuse the existing id if this (pair, target_time) was already stored.
    existing = db.predictions.find_one(
        {"pair": pair, "target_time": target_time}, {"id": 1}
    )
    new_id = existing["id"] if existing else db.next_id("prediction")
    record["id"] = new_id

    db.predictions.update_one(
        {"pair": pair, "target_time": target_time},
        {"$set": record},
        upsert=True,
    )
    return new_id


def update_actuals(record_id: int, actual_upper_open: float, actual_lower_open: float, ohlc: dict):
    """Fill actual pip moves + the closing price for a previously stored prediction."""
    result = db.predictions.update_one(
        {"id": record_id},
        {"$set": {
            "actual_upper_open": actual_upper_open,
            "actual_lower_open": actual_lower_open,
            "live_close": ohlc["close"],
        }},
    )
    if result.matched_count == 0:
        logger.warning("update_actuals: prediction id=%s not found", record_id)


def backfill_actuals(pair: str, pip_div: float = 10000.0) -> int:
    """
    Fill actuals for any prediction whose target candle has since closed.

    Self-healing: matches `predictions.target_time` to an `ohlcv.time` candle.
    The existence of that candle is the signal that it has closed — no wall-clock
    comparison is used, because the broker feed's candle timestamps may be offset
    from real UTC (and would otherwise never satisfy `target_time <= now`).
    Predictions whose target candle never materializes (e.g. the last candle
    before a weekend close) simply stay unfilled — which is correct.
    Returns the number of predictions filled this call.
    """
    pending = db.predictions.find({"pair": pair, "actual_upper_close": None})
    filled = 0
    for p in pending:
        candle = db.ohlcv.find_one({"pair": pair, "time": p["target_time"]})
        if not candle:
            continue
        # The target candle must be fully closed before we read its high/low.
        # save_ohlcv upserts the live/forming candle too, so its mere existence
        # is not enough — a strictly-later candle must exist. Otherwise we'd fill
        # actuals from an incomplete candle and, since `pending` only selects
        # actual_upper_close == None, never recompute them (frozen/wrong values).
        newer = db.ohlcv.find_one({"pair": pair, "time": {"$gt": p["target_time"]}})
        if not newer:
            continue
        # Close baseline = close of the candle the prediction was made from.
        # Stored on the record since the schema change; fall back to the ohlcv
        # candle at current_candle_time for older predictions.
        base_close = p.get("base_close")
        if base_close is None:
            base_candle = db.ohlcv.find_one({"pair": pair, "time": p.get("current_candle_time")})
            if not base_candle:
                continue
            base_close = base_candle["close"]
        no, nh, nl = candle["open"], candle["high"], candle["low"]
        db.predictions.update_one(
            {"_id": p["_id"]},
            {"$set": {
                # open-baseline (vs next-candle open)
                "actual_upper_open": (nh - no) * pip_div,
                "actual_lower_open": (no - nl) * pip_div,
                # close-baseline (vs current candle close / base_close)
                "actual_upper_close": (nh - base_close) * pip_div,
                "actual_lower_close": (base_close - nl) * pip_div,
                "live_close": float(candle["close"]),
            }},
        )
        filled += 1
    return filled


def get_latest_prediction() -> dict | None:
    doc = db.predictions.find_one(sort=[("id", -1)])
    return _clean(doc) if doc else None


def get_recent_predictions(pair: str, limit: int = 100) -> list[dict]:
    """Most recent `limit` predictions for a pair, newest first."""
    cursor = (
        db.predictions.find({"pair": pair}, {"_id": 0})
        .sort("id", -1)
        .limit(limit)
    )
    return list(cursor)


def get_accuracy_stats() -> dict:
    """
    Coverage stats over evaluated predictions. Coverage is computed on the fly
    from the actual pip move vs the predicted q10–q90 pip range.
    """
    query = {"actual_upper_open": {"$ne": None}, "actual_lower_open": {"$ne": None}}
    evaluated = list(db.predictions.find(query, {
        "_id": 0, "actual_upper_open": 1, "actual_lower_open": 1,
        "upper_q10_pip": 1, "upper_q90_pip": 1,
        "lower_q10_pip": 1, "lower_q90_pip": 1,
    }))
    total = len(evaluated)
    if total == 0:
        return {
            "total_evaluated": 0,
            "upper_coverage_pct": 0.0,
            "lower_coverage_pct": 0.0,
            "both_covered_pct": 0.0,
        }
    upper = lower = both = 0
    for d in evaluated:
        uc = (
            d.get("upper_q10_pip") is not None
            and d["upper_q10_pip"] <= d["actual_upper_open"] <= d["upper_q90_pip"]
        )
        lc = (
            d.get("lower_q10_pip") is not None
            and d["lower_q10_pip"] <= d["actual_lower_open"] <= d["lower_q90_pip"]
        )
        upper += uc
        lower += lc
        both += uc and lc
    return {
        "total_evaluated": total,
        "upper_coverage_pct": round(upper / total * 100, 2),
        "lower_coverage_pct": round(lower / total * 100, 2),
        "both_covered_pct": round(both / total * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live data sources (OHLCV / GDELT / Forex Factory)
# ─────────────────────────────────────────────────────────────────────────────
def save_ohlcv(pair: str, df: pd.DataFrame, tail: int = 5):
    """Upsert the most recent `tail` candles, keyed by (pair, time)."""
    if df is None or df.empty:
        return
    ops = []
    for ts, row in df.tail(tail).iterrows():
        time = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        ops.append(UpdateOne(
            {"pair": pair, "time": time},
            {"$set": {
                "pair": pair,
                "time": time,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }},
            upsert=True,
        ))
    if ops:
        db.ohlcv.bulk_write(ops, ordered=False)


def save_gdelt(batch_time: datetime | None, scores: dict):
    """Upsert one row per currency for a GDELT 15-min batch."""
    if batch_time is None or not scores:
        return
    ops = []
    for currency, vals in scores.items():
        ops.append(UpdateOne(
            {"batch_time": batch_time, "currency": currency},
            {"$set": {
                "batch_time": batch_time,
                "currency": currency,
                "ws_z": float(vals.get("ws_z", 0.0)),
                "ma_z": float(vals.get("ma_z", 0)),
            }},
            upsert=True,
        ))
    if ops:
        db.gdelt.bulk_write(ops, ordered=False)


def save_news_hourly(hour: datetime, scores: dict):
    """
    Upsert one row per currency for a 1H news bucket.

    `scores` maps currency → {combined_z, ma_z, ws_z, sf_z}. These are the
    per-hour LSTM inputs (combined_z = GDELT ws_z + FF surprise), persisted so
    the encoder can be fed a real 24-hour window instead of a repeated value.
    Keyed by (hour, currency); re-running within the same hour overwrites.
    """
    if hour is None or not scores:
        return
    ops = []
    for currency, vals in scores.items():
        ops.append(UpdateOne(
            {"hour": hour, "currency": currency},
            {"$set": {
                "hour": hour,
                "currency": currency,
                "combined_z": float(vals.get("combined_z", 0.0)),
                "ma_z": float(vals.get("ma_z", 0.0)),
                "ws_z": float(vals.get("ws_z", 0.0)),
                "sf_z": float(vals.get("sf_z", 0.0)),
            }},
            upsert=True,
        ))
    if ops:
        db.news_hourly.bulk_write(ops, ordered=False)


def get_news_stats(currency: str) -> dict | None:
    """Return the running expanding-z-score stats for a currency, or None."""
    return db.news_stats.find_one({"currency": currency}, {"_id": 0})


def save_news_stats(currency: str, stats: dict):
    """Upsert the running {wm:{n,mean,M2}, ma:{n,mean,M2}} stats for a currency."""
    db.news_stats.update_one(
        {"currency": currency},
        {"$set": {"currency": currency, **stats}},
        upsert=True,
    )


def count_news_hourly(currency: str) -> int:
    """How many hourly news rows are stored for a currency (cold-start check)."""
    return db.news_hourly.count_documents({"currency": currency})


def get_recent_news_hourly(currency: str, limit: int = 24) -> list[dict]:
    """
    Most recent `limit` hourly news rows for a currency, oldest-first.

    Returns ascending by hour so the caller can feed it straight into the
    LSTM sliding window (last timestep = newest hour).
    """
    cursor = (
        db.news_hourly.find({"currency": currency}, {"_id": 0})
        .sort("hour", -1)
        .limit(limit)
    )
    rows = list(cursor)
    rows.reverse()   # newest-first → oldest-first
    return rows


def get_ohlcv(pair: str, limit: int = 100) -> list[dict]:
    """Most recent `limit` candles for a pair, newest first."""
    cursor = (
        db.ohlcv.find({"pair": pair}, {"_id": 0})
        .sort("time", -1)
        .limit(limit)
    )
    return list(cursor)


OUT_TZ = timezone(timedelta(hours=8))


def _to_bkk(dt: datetime | None) -> datetime | None:
    """Convert naive UTC datetime to +08:00 timezone-aware datetime."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(OUT_TZ)


def iter_predictions_with_candles(
    pair: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
):
    match = {}
    if pair:
        match["pair"] = pair
    if start or end:
        rng = {}
        if start:
            rng["$gte"] = start
        if end:
            rng["$lt"] = end
        match["current_candle_time"] = rng

    pipeline = [
        {"$match": match},
        {"$lookup": {
            "from": "ohlcv",
            "let": {"p": "$pair", "t": "$current_candle_time"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [
                    {"$eq": ["$pair", "$$p"]},
                    {"$eq": ["$time", "$$t"]},
                ]}}},
                {"$project": {"_id": 0, "open": 1, "high": 1, "low": 1, "close": 1}},
            ],
            "as": "cur",
        }},
        {"$lookup": {
            "from": "ohlcv",
            "let": {"p": "$pair", "t": "$target_time"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [
                    {"$eq": ["$pair", "$$p"]},
                    {"$eq": ["$time", "$$t"]},
                ]}}},
                {"$project": {"_id": 0, "open": 1, "high": 1, "low": 1, "close": 1}},
            ],
            "as": "nxt",
        }},
        {"$unwind": {"path": "$cur", "preserveNullAndEmptyArrays": True}},
        {"$unwind": {"path": "$nxt", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "_id": 0,
            "current_candle_time": 1,
            "target_time": 1,
            "actual_upper_open": 1,
            "actual_lower_open": 1,
            "actual_upper_close": 1,
            "actual_lower_close": 1,
            "upper_q10_pip": 1,
            "upper_q50_pip": 1,
            "upper_q90_pip": 1,
            "lower_q10_pip": 1,
            "lower_q50_pip": 1,
            "lower_q90_pip": 1,
            "current_open": "$cur.open",
            "current_high": "$cur.high",
            "current_low": "$cur.low",
            "current_close": "$cur.close",
            "upper_q10_price": 1,
            "lower_q10_price": 1,
            "upper_q50_price": 1,
            "lower_q50_price": 1,
            "upper_q90_price": 1,
            "lower_q90_price": 1,
            "next_candle_open": "$nxt.open",
            "next_candle_high": "$nxt.high",
            "next_candle_low": "$nxt.low",
            "next_candle_close": "$nxt.close",
        }},
        {"$sort": {"current_candle_time": 1}},
    ]

    for row in db.predictions.aggregate(pipeline):
        row["time"] = _to_bkk(row.get("current_candle_time"))
        row["target_time"] = _to_bkk(row.get("target_time"))
        yield row


def save_gdelt_raw(records: list[dict]):
    """Upsert currency-relevant raw GDELT events, keyed by global_event_id."""
    if not records:
        return
    ops = []
    for r in records:
        gid = r.get("global_event_id")
        if gid is None or pd.isna(gid):
            continue
        doc = {
            "global_event_id": int(gid),
            "batch_time": r.get("batch_time"),
            "currency": r.get("currency"),
            "day": _num(r.get("day")),
            "actor1_country": _str(r.get("actor1_country")),
            "actor2_country": _str(r.get("actor2_country")),
            "event_code": _str(r.get("event_code")),
            "goldstein": _num(r.get("goldstein")),
            "num_mentions": _num(r.get("num_mentions")),
            "num_sources": _num(r.get("num_sources")),
            "num_articles": _num(r.get("num_articles")),
            "avg_tone": _num(r.get("avg_tone")),
        }
        ops.append(UpdateOne(
            {"global_event_id": doc["global_event_id"]},
            {"$set": doc},
            upsert=True,
        ))
    if ops:
        db.gdelt_raw.bulk_write(ops, ordered=False)


def save_ff_events(df: pd.DataFrame):
    """Upsert calendar events, keyed by (event_time, country, title)."""
    if df is None or df.empty:
        return
    ops = []
    for _, row in df.iterrows():
        event_time = row["date"]
        if hasattr(event_time, "to_pydatetime"):
            event_time = event_time.to_pydatetime()
        title = str(row.get("title", "")) or ""
        ops.append(UpdateOne(
            {"event_time": event_time, "country": row["country"], "title": title},
            {"$set": {
                "event_time": event_time,
                "country": row["country"],
                "title": title,
                "impact": row.get("impact"),
                "forecast": _num(row.get("forecast_val")),
                "actual": _num(row.get("actual_val")),
                "surprise": float(row.get("Surprise_Factor", 0.0)),
            }},
            upsert=True,
        ))
    if ops:
        db.ff_events.bulk_write(ops, ordered=False)


def _num(v):
    """Coerce to float or None (NaN-safe) for storage."""
    if v is None or pd.isna(v):
        return None
    return float(v)


def _str(v):
    """Coerce to stripped str or None (NaN-safe) for storage."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None
