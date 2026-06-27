"""
api/db.py
MongoDB client, collection handles, and index setup.

Collections (natural keys → upsert handles dedup across the 5-min cycle):
  ohlcv       (pair, time)                  — 5-min candles
  gdelt       (batch_time, currency)        — 15-min aggregated sentiment
  gdelt_raw   (global_event_id)             — raw currency-relevant GDELT events
  ff_events   (event_time, country, title)  — irregular calendar events
  news_hourly (hour, currency)              — 1H GDELT+FF sentiment for the LSTM window
  news_stats  (currency)                    — running WM/MA stats for expanding z-score
  predictions (pair, target_time)           — model output per future candle
  counters    auto-increment ids
"""
import logging
from pymongo import MongoClient, ASCENDING, ReturnDocument

from api.config import MONGO_URI, MONGO_DB

logger = logging.getLogger(__name__)

_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = _client[MONGO_DB]

ohlcv = db["ohlcv"]
gdelt = db["gdelt"]
gdelt_raw = db["gdelt_raw"]
ff_events = db["ff_events"]
news_hourly = db["news_hourly"]
news_stats = db["news_stats"]
predictions = db["predictions"]
counters = db["counters"]


def init_indexes():
    """Create unique indexes. Idempotent — safe to call on every startup."""
    ohlcv.create_index([("pair", ASCENDING), ("time", ASCENDING)], unique=True)
    gdelt.create_index([("batch_time", ASCENDING), ("currency", ASCENDING)], unique=True)
    gdelt_raw.create_index([("global_event_id", ASCENDING)], unique=True)
    gdelt_raw.create_index([("batch_time", ASCENDING)])
    ff_events.create_index(
        [("event_time", ASCENDING), ("country", ASCENDING), ("title", ASCENDING)],
        unique=True,
    )
    news_hourly.create_index([("hour", ASCENDING), ("currency", ASCENDING)], unique=True)
    news_hourly.create_index([("currency", ASCENDING), ("hour", ASCENDING)])
    news_stats.create_index([("currency", ASCENDING)], unique=True)
    predictions.create_index([("pair", ASCENDING), ("target_time", ASCENDING)], unique=True)
    predictions.create_index([("id", ASCENDING)], unique=True)
    logger.info("MongoDB indexes ensured on db '%s'", MONGO_DB)


def next_id(name: str = "prediction") -> int:
    """Atomic auto-increment counter (one sequence per name)."""
    doc = counters.find_one_and_update(
        {"_id": name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(doc["seq"])


def ping() -> bool:
    """Return True if the server responds."""
    try:
        _client.admin.command("ping")
        return True
    except Exception as e:
        logger.warning("MongoDB ping failed: %s", e)
        return False
