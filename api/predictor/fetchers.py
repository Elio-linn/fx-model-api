"""
api/predictor/fetchers.py
Live data fetchers: OHLCV (external API), GDELT sentiment, Forex Factory calendar.
"""
import io
import logging
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

from api.config import CURRENCIES, PAIR, OHLCV_API_BASE

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV
# ─────────────────────────────────────────────────────────────────────────────
def get_live_ohlcv(pair: str = PAIR, count: int = 500) -> pd.DataFrame | None:
    """Fetch recent 5-minute OHLCV candles from the external OHLC API."""
    try:
        resp = requests.get(
            f"{OHLCV_API_BASE}/ohlc/{pair}",
            params={"count": count},
            timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json().get("data") or []
    except Exception as e:
        logger.warning("candle missing: API request failed for %s: %s", pair, e)
        return None

    if not rows:
        logger.warning("candle missing: API returned no candles for %s", pair)
        return None

    df = pd.DataFrame(rows)
    if df.empty or "time" not in df.columns:
        logger.warning("candle missing: malformed candle data for %s", pair)
        return None

    # The broker feed timestamps are in the broker server timezone (a whole-hour
    # offset from UTC, e.g. GMT+3), not real UTC. Detect that offset from the most
    # recent candle vs wall-clock UTC, round to the nearest hour, and shift so the
    # index becomes true UTC. (The model is timezone-agnostic — it uses no calendar
    # features — so this only corrects labels and keeps ohlcv.time aligned with
    # GDELT/FF and the prediction timestamps.)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    offset_hours = round((df.index[-1] - now_naive).total_seconds() / 3600)
    if offset_hours:
        df.index = df.index - pd.Timedelta(hours=offset_hours)
        logger.debug("OHLCV broker offset %+d h → shifted to UTC", offset_hours)
    df.index = df.index.tz_localize("UTC")
    df.index.name = "DateTime_UTC"

    # Normalize to the OHLCV column schema the pipeline expects
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.tail(count)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def check_data_source(pair: str = PAIR, timeout: int = 5) -> bool:
    """
    Lightweight reachability check for the OHLC API.
    Returns True if the API responds 200 with at least one candle.
    """
    try:
        resp = requests.get(
            f"{OHLCV_API_BASE}/ohlc/{pair}",
            params={"count": 1},
            timeout=timeout,
        )
        return resp.status_code == 200 and bool(resp.json().get("data"))
    except Exception as e:
        logger.warning("data source unreachable for %s: %s", pair, e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GDELT Sentiment
# ─────────────────────────────────────────────────────────────────────────────
def _gdelt_batch_time(latest_url: str) -> datetime:
    """Extract the 15-min batch timestamp (YYYYMMDDHHMMSS) from a GDELT file URL."""
    stem = latest_url.rsplit("/", 1)[-1][:14]  # e.g. 20260612101500
    return datetime.strptime(stem, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


# Per-currency GDELT actor country codes (Actor1/2CountryCode, ISO 3-letter;
# "EUR" = supranational Europe).
GDELT_COUNTRY_CODES = {
    "USD": ["USA"],
    "EUR": ["FRA", "DEU", "ITA", "EUR"],
}


# GDELT 2.0 Event file columns (0-indexed) → our field names.
GDELT_COLS = {
    0:  "global_event_id",
    1:  "day",
    7:  "actor1_country",
    17: "actor2_country",
    26: "event_code",
    30: "goldstein",
    31: "num_mentions",
    32: "num_sources",
    33: "num_articles",
    34: "avg_tone",
}


def get_live_gdelt() -> tuple[dict, datetime | None, list[dict]]:
    """
    Fetch the latest GDELT 2.0 event batch and compute sentiment per currency.
    Returns (scores, batch_time, raw_records):
      - scores: {currency: {ws_z, ma_z}} aggregate for the model
      - batch_time: GDELT's own 15-min slot
      - raw_records: currency-relevant raw event rows (for gdelt_raw collection)
    """
    batch_time = None
    try:
        url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        latest_url = resp.text.strip().split("\n")[0].split(" ")[2]
        batch_time = _gdelt_batch_time(latest_url)

        resp = requests.get(latest_url, headers=headers, timeout=30)
        resp.raise_for_status()

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_filename = zf.namelist()[0]

        # Read by position, then rename (avoids the usecols/names ordering trap).
        df = pd.read_csv(
            zf.open(csv_filename), sep="\t", header=None,
            usecols=list(GDELT_COLS.keys()), low_memory=False,
        ).rename(columns=GDELT_COLS)

        df["goldstein"] = pd.to_numeric(df["goldstein"], errors="coerce").fillna(0.0)
        df["avg_tone"] = pd.to_numeric(df["avg_tone"], errors="coerce").fillna(0.0)

        res = {}
        raw_records = []
        for cur in CURRENCIES:
            actor_codes = GDELT_COUNTRY_CODES.get(cur, [])
            filtered = df[
                df["actor1_country"].isin(actor_codes) |
                df["actor2_country"].isin(actor_codes)
            ]
            if not filtered.empty:
                res[cur] = {
                    "ws_z": float(filtered["goldstein"].mean() + filtered["avg_tone"].mean()),
                    "ma_z": len(filtered),
                }
                for _, r in filtered.iterrows():
                    rec = {col: r[col] for col in GDELT_COLS.values()}
                    rec["currency"] = cur
                    rec["batch_time"] = batch_time
                    raw_records.append(rec)
            else:
                res[cur] = {"ws_z": 0.0, "ma_z": 0}
        return res, batch_time, raw_records

    except Exception as e:
        logger.warning("GDELT error: %s", e)
        return {cur: {"ws_z": 0.0, "ma_z": 0} for cur in CURRENCIES}, batch_time, []


_GDELT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_live_gdelt_hourly(n_batches: int = 4) -> tuple[dict, datetime | None, list[dict]]:
    """
    Fetch the last `n_batches` GDELT 2.0 15-min batches (the trailing hour) and
    aggregate per currency exactly the way phase1_gdelt_aggregator.py trained on:

        True_Sentiment    = AvgTone × GoldsteinScale × total_mentions   (per event)
        Weighted_Momentum = Σ True_Sentiment / Σ total_mentions          (per hour)
        Market_Attention  = Σ total_mentions                             (per hour)

    Cleaning matches phase1: GoldsteinScale/AvgTone clipped to [-10, 10],
    total_mentions filled→1 and clipped to [1, 1000]. Returns the *raw* hourly
    Weighted_Momentum / Market_Attention; the expanding z-score (ws_z/ma_z) is
    applied downstream against running stats seeded from the training parquet.

    Returns (stats, batch_time, raw_records):
      stats:       {currency: {"wm": float, "ma": float, "n": int}}
      batch_time:  the latest 15-min slot (used as the gdelt collection key)
      raw_records: currency-relevant raw rows across the fetched batches
    """
    stats = {cur: {"wm": 0.0, "ma": 0.0, "n": 0} for cur in CURRENCIES}
    batch_time = None
    try:
        resp = requests.get(
            "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
            headers=_GDELT_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        latest_url = resp.text.strip().split("\n")[0].split(" ")[2]
        batch_time = _gdelt_batch_time(latest_url)
    except Exception as e:
        logger.warning("GDELT lastupdate error: %s", e)
        return stats, None, []

    # Build the n preceding 15-min batch URLs from the latest one.
    fname = latest_url.rsplit("/", 1)[-1]          # 20260616031500.export.CSV.zip
    base_url = latest_url.rsplit("/", 1)[0]
    suffix = fname[14:]                            # .export.CSV.zip

    frames = []
    for k in range(n_batches):
        bt = batch_time - timedelta(minutes=15 * k)
        url = f"{base_url}/{bt.strftime('%Y%m%d%H%M%S')}{suffix}"
        try:
            r = requests.get(url, headers=_GDELT_HEADERS, timeout=30)
            r.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(
                zf.open(zf.namelist()[0]), sep="\t", header=None,
                usecols=list(GDELT_COLS.keys()), low_memory=False,
            ).rename(columns=GDELT_COLS)
            frames.append(df)
        except Exception as e:
            logger.warning("GDELT batch %s fetch failed: %s", bt, e)

    if not frames:
        return stats, batch_time, []

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["global_event_id"])
    df["goldstein"]    = pd.to_numeric(df["goldstein"], errors="coerce").fillna(0.0).clip(-10.0, 10.0)
    df["avg_tone"]     = pd.to_numeric(df["avg_tone"], errors="coerce").fillna(0.0).clip(-10.0, 10.0)
    df["num_mentions"] = pd.to_numeric(df["num_mentions"], errors="coerce").fillna(1.0).clip(1.0, 1000.0)
    df["true_sentiment"] = df["avg_tone"] * df["goldstein"] * df["num_mentions"]

    raw_records = []
    for cur in CURRENCIES:
        actor_codes = GDELT_COUNTRY_CODES.get(cur, [])
        filtered = df[
            df["actor1_country"].isin(actor_codes) |
            df["actor2_country"].isin(actor_codes)
        ]
        if filtered.empty:
            continue
        ts_sum = float(filtered["true_sentiment"].sum())
        m_sum = float(filtered["num_mentions"].sum())
        stats[cur] = {
            "wm": ts_sum / m_sum if m_sum > 0 else 0.0,   # Weighted_Momentum
            "ma": m_sum,                                   # Market_Attention
            "n": int(len(filtered)),
        }
        for _, r in filtered.iterrows():
            rec = {col: r[col] for col in GDELT_COLS.values()}
            rec["currency"] = cur
            rec["batch_time"] = batch_time
            raw_records.append(rec)

    return stats, batch_time, raw_records


# ─────────────────────────────────────────────────────────────────────────────
# Forex Factory Calendar
# ─────────────────────────────────────────────────────────────────────────────
def _clean_ff_value(val_str) -> float | None:
    if pd.isna(val_str) or not str(val_str).strip():
        return None
    val_str = str(val_str).replace("<", "").replace(">", "").strip()
    mult = 1.0
    val_upper = val_str.upper()
    if "K" in val_upper:
        mult = 1e3; val_str = val_upper.replace("K", "")
    elif "M" in val_upper:
        mult = 1e6; val_str = val_upper.replace("M", "")
    elif "B" in val_upper:
        mult = 1e9; val_str = val_upper.replace("B", "")
    elif "%" in val_str:
        val_str = val_str.replace("%", "")
    try:
        return float(val_str) * mult
    except ValueError:
        return None


# Forex Factory calendar via the official weekly JSON feed.
# The HTML calendar at forexfactory.com sits behind Cloudflare, which 403s
# datacenter IPs (e.g. AWS EC2) even with cloudscraper. faireconomy.media is the
# same data ForexFactory publishes for its own apps/widgets — plain JSON, no
# Cloudflare, reachable from any host. Re-fetching fills `actual` as events
# publish (self-healing within the week).
_FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_live_ff() -> pd.DataFrame:
    """
    Fetch THIS WEEK's Forex Factory calendar from the official JSON feed.
    Returns High/Medium EUR/USD events: [date, country, title, impact,
    forecast_val, actual_val, Surprise_Factor]. `date` is tz-aware UTC.
    """
    try:
        resp = requests.get(_FF_URL, headers=_FF_HEADERS, timeout=30)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        logger.warning("ForexFactory fetch failed: %s", e)
        return pd.DataFrame()

    cols = ["date", "country", "title", "impact",
            "forecast_val", "actual_val", "Surprise_Factor"]
    rows = []
    for ev in events:
        country = ev.get("country")
        impact = ev.get("impact")
        title = ev.get("title") or ""
        if country not in CURRENCIES or impact not in ("High", "Medium") or not title:
            continue

        # date is ISO 8601 with offset, e.g. "2026-06-17T14:00:00-04:00".
        try:
            event_dt = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
        except (KeyError, ValueError, TypeError):
            continue

        fv = _clean_ff_value(ev.get("forecast"))
        av = _clean_ff_value(ev.get("actual"))   # absent until the event publishes
        surprise = (av - fv) if (av is not None and fv is not None) else 0.0
        rows.append({
            "date": event_dt, "country": country, "title": title, "impact": impact,
            "forecast_val": fv, "actual_val": av, "Surprise_Factor": surprise,
        })

    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df
