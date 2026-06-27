"""
api/scheduler.py
APScheduler-based 5-minute aligned background job.
The job is identical in logic to the main loop in live_backtest_tft.py
but runs as a non-blocking background thread managed by FastAPI lifespan.
"""
import sys
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from api import state, storage
from api.predictor.fetchers import get_live_ohlcv, get_live_gdelt_hourly, get_live_ff, check_data_source
from api.predictor.features import run_live_prediction, compute_latents, LSTM_WINDOW_HOURS
from api.predictor.model_loader import (
    get_tft, get_lstm_encoders, get_feature_config, get_ref_dataset
)
from api.predictor import sentiment
from api.config import BASE_DIR, PAIR, CURRENCIES, pip_divisor

logger = logging.getLogger(__name__)

# Ensure project root is importable (for train_tft.build_dataset)
_root = str(BASE_DIR)
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── Hourly news / sentiment ──────────────────────────────────────────────────
# GDELT, Forex Factory, and the LSTM sentiment encoding are refreshed once per
# hour (not every 5-min cycle). The model was trained on 1H news merge_asof'd
# into 5M candles, so the news latents only change hourly — the TFT itself still
# predicts every cycle, reusing the cached latents. The encoder is fed a real
# 24-hour [combined_z, ma_z] window assembled from the news_hourly collection.
_news_cache = {"hour": None, "latents": None, "ff_aggregates": None}


def _compute_hourly_news(hour_utc: datetime, now_utc: datetime) -> tuple[dict, dict]:
    """
    Fetch GDELT + Forex Factory once for this hour, persist the 1H sentiment
    bucket, then encode a real 24h sliding window into per-currency LSTM latents.

    GDELT is aggregated over the trailing hour (4×15-min batches) into
    Weighted_Momentum / Market_Attention, then expanding-z-scored against running
    stats seeded from training (phase1 parity). channel-0 of the encoder input is
    combined_z = clip(ws_z + FF surprise, ±10); channel-1 is the z-scored ma_z.
    """
    gdelt_stats, gdelt_batch, gdelt_raw = get_live_gdelt_hourly()
    ff_data = get_live_ff()

    # Persist raw sources (hourly cadence now).
    storage.save_gdelt_raw(gdelt_raw)
    storage.save_ff_events(ff_data)

    scores = {}        # → news_hourly (LSTM window)
    gdelt_scores = {}  # → gdelt collection (z-scored ws_z/ma_z)
    ff_by_cur = {}     # per-currency clean FF aggregates for this hour
    for cur in CURRENCIES:
        st = storage.get_news_stats(cur) or {}
        wm = float(gdelt_stats[cur]["wm"])
        ma = float(gdelt_stats[cur]["ma"])
        # expanding z-score (uses prior stats), then fold this hour in
        ws_z, st_wm = sentiment.zscore_update(st.get("wm", {}), wm)
        ma_z, st_ma = sentiment.zscore_update(st.get("ma", {}), ma)
        # phase2 clips ws_z/ma_z to [-10, 10] before encoding
        ws_z = max(-10.0, min(10.0, ws_z))
        ma_z = max(-10.0, min(10.0, ma_z))
        storage.save_news_stats(cur, {"wm": st_wm, "ma": st_ma})

        # FF events released in the trailing hour for this currency.
        sf = 0.0
        ev_count = 0
        has_high = 0
        if not ff_data.empty:
            recent_ff = ff_data[
                (ff_data["country"] == cur) &
                (ff_data["date"] > now_utc - timedelta(hours=1))
            ]
            if not recent_ff.empty:
                sf = float(recent_ff["Surprise_Factor"].sum())   # sparse; raw → combined
                ev_count = int(len(recent_ff))
                has_high = int((recent_ff["impact"] == "High").any())
        combined = max(-10.0, min(10.0, ws_z + sf))
        scores[cur] = {"combined_z": combined, "ma_z": ma_z, "ws_z": ws_z, "sf_z": sf}
        gdelt_scores[cur] = {"ws_z": ws_z, "ma_z": ma_z}
        ff_by_cur[cur] = {"event_count": ev_count, "has_high_impact": has_high}

    storage.save_gdelt(gdelt_batch, gdelt_scores)
    storage.save_news_hourly(hour_utc, scores)

    # Map currencies → base/quote for the TFT FF feature columns.
    base_cur, quote_cur = PAIR[:3], PAIR[3:6]
    ff_aggregates = {
        "base": ff_by_cur.get(base_cur, {"event_count": 0, "has_high_impact": 0}),
        "quote": ff_by_cur.get(quote_cur, {"event_count": 0, "has_high_impact": 0}),
    }

    # Build a real 24h sliding window per currency from stored hourly rows,
    # left-padding with zeros until 24 hours of history have accumulated.
    windows = {}
    for cur in CURRENCIES:
        rows = storage.get_recent_news_hourly(cur, LSTM_WINDOW_HOURS)   # oldest-first
        seq = [[r["combined_z"], r["ma_z"]] for r in rows]
        pad = LSTM_WINDOW_HOURS - len(seq)
        if pad > 0:
            seq = [[0.0, 0.0]] * pad + seq
        windows[cur] = np.asarray(seq, dtype=np.float32)

    latents = compute_latents(windows, get_lstm_encoders())
    logger.info(
        "Hourly news refreshed for %s (GDELT+FF → 24h LSTM window, %d currencies)",
        hour_utc.strftime("%Y-%m-%d %H:00"), len(CURRENCIES),
    )
    return latents, ff_aggregates


def _get_hourly_news(now_utc: datetime) -> tuple[dict, dict]:
    """Return cached (latents, ff_aggregates), recomputing only when the UTC hour rolls over."""
    hour_utc = now_utc.replace(minute=0, second=0, microsecond=0)
    if _news_cache["hour"] != hour_utc or _news_cache["latents"] is None:
        latents, ff_aggregates = _compute_hourly_news(hour_utc, now_utc)
        _news_cache["latents"] = latents
        _news_cache["ff_aggregates"] = ff_aggregates
        _news_cache["hour"] = hour_utc
    return _news_cache["latents"], _news_cache["ff_aggregates"]


def is_market_open(now_utc: datetime | None = None) -> bool:
    """
    FX market hours: open Sun ~22:00 UTC → close Fri ~22:00 UTC.
    Closed all of Saturday, Sunday before 22:00, and Friday after 22:00.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    wd = now_utc.weekday()  # Mon=0 ... Sat=5, Sun=6
    if wd == 5:                              # Saturday — closed all day
        return False
    if wd == 6 and now_utc.hour < 22:        # Sunday before open
        return False
    if wd == 4 and now_utc.hour >= 22:       # Friday after close
        return False
    return True


def _run_cycle():
    """
    One 5-minute cycle:
    1. Fetch live data
    2. Update actuals for previous prediction
    3. Run TFT prediction
    4. Store new prediction row (actuals pending)
    """
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now().astimezone()
    logger.info("Cycle start: %s", now_local)

    state.set_last_run(now_utc)

    # ── 0. Market hours check ──────────────────────────────────────────────
    # Weekends have no FX data — skip without counting it as an error.
    if not is_market_open(now_utc):
        # Market is closed, but still refresh data-source health for the UI.
        state.set_model_status("active" if check_data_source() else "inactive")
        logger.info("Market closed (weekend). Skipping cycle.")
        state.set_status("market_closed", "Market closed (weekend)", market_open=False)
        return

    try:
        # ── 1. Fetch OHLCV (every cycle) ───────────────────────────────────
        full_history = get_live_ohlcv(PAIR, count=500)

        if full_history is None or full_history.empty:
            # Distinguish API down (inactive) from API up but empty (active).
            state.set_model_status("active" if check_data_source() else "inactive")
            logger.warning("Candle missing: failed to fetch OHLCV. Skipping cycle.")
            state.set_status("candle_missing", "No candle data from source", market_open=True)
            state.increment_error()
            return

        # Fetch succeeded → data source is reachable.
        state.set_model_status("active")

        # ── 1b. Persist candles (upsert; dedupes across cycles) ────────────
        storage.save_ohlcv(PAIR, full_history)

        # ── 1c. Hourly news + LSTM latents (cached across the hour) ────────
        # GDELT/FF fetch + LSTM encoding only run when the UTC hour rolls over;
        # the 5-min cycles in between reuse the cached latents + FF aggregates.
        latents, ff_aggregates = _get_hourly_news(now_utc)

        current_time_utc = full_history.index[-1]
        current_candle = full_history.iloc[-1]
        state.set_last_candle(str(current_time_utc), current_candle)

        # ── 2. Backfill actuals for any prediction whose target candle closed ─
        # DB-driven (target_time ↔ ohlcv.time): self-healing across restarts and
        # weekend gaps, unlike the old in-memory "fill the previous row" logic.
        filled = storage.backfill_actuals(PAIR, pip_divisor(PAIR))
        if filled:
            logger.info("Backfilled actuals for %d prediction(s)", filled)

        # ── 3. TFT Prediction ───────────────────────────────────────────────
        cfg = get_feature_config()
        tft = get_tft()
        ref_ds = get_ref_dataset()

        tft_input = run_live_prediction(full_history, latents, cfg, ff_aggregates)
        if tft_input is None:
            logger.warning("Insufficient history for TFT. Skipping.")
            state.set_status("candle_missing", "Insufficient history for prediction", market_open=True)
            return

        from train_tft import build_dataset
        ds = build_dataset(tft_input, cfg, reference_dataset=ref_ds, predict=True)
        dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
        preds = tft.predict(dl, mode="quantiles", return_y=False)

        out_upper = preds[0].cpu().numpy()[0, 0, :]
        out_lower = preds[1].cpu().numpy()[0, 0, :]

        target_time_utc = current_time_utc + timedelta(minutes=5)
        target_time_local = target_time_utc.to_pydatetime().astimezone()

        # Wall-clock time the prediction result was produced (+ its 5-min expiry).
        prediction_start_utc = datetime.now(timezone.utc).replace(microsecond=0)
        prediction_expiry_utc = prediction_start_utc + timedelta(minutes=5)

        logger.info(
            "Predicted for %s → Upper Q50: %.1f | Lower Q50: %.1f",
            target_time_local.strftime("%H:%M"), out_upper[1], out_lower[1],
        )
        state.set_status("ok", "Running", market_open=True)

        # ── 4. Store new prediction ─────────────────────────────────────────
        candle_close = float(current_candle["Close"])
        pip_div = pip_divisor(PAIR)   # 10000 for non-JPY, 100 for JPY pairs
        record = {
            "current_candle_time": current_time_utc.to_pydatetime(),
            "prediction_start_time": prediction_start_utc,
            "prediction_expiry_time": prediction_expiry_utc,
            "target_time": target_time_utc.to_pydatetime(),
            "pair": PAIR,
            "timeframe": "5M",
            # Predicted range in pips
            "upper_q10_pip": float(out_upper[0]),
            "upper_q50_pip": float(out_upper[1]),
            "upper_q90_pip": float(out_upper[2]),
            "lower_q10_pip": float(out_lower[0]),
            "lower_q50_pip": float(out_lower[1]),
            "lower_q90_pip": float(out_lower[2]),
            # Predicted range as price levels (candle close ± pip/divisor)
            "upper_q10_price": candle_close + (float(out_upper[0]) / pip_div),
            "upper_q50_price": candle_close + (float(out_upper[1]) / pip_div),
            "upper_q90_price": candle_close + (float(out_upper[2]) / pip_div),
            "lower_q10_price": candle_close - (float(out_lower[0]) / pip_div),
            "lower_q50_price": candle_close - (float(out_lower[1]) / pip_div),
            "lower_q90_price": candle_close - (float(out_lower[2]) / pip_div),
            # Actuals (filled once the target candle closes)
            "actual_upper": None,
            "actual_lower": None,
            "live_close": None,
        }
        new_id = storage.append_prediction(record)
        state.set_last_prediction_id(new_id)

    except Exception as e:
        logger.exception("Error in cycle: %s", e)
        state.set_status("error", str(e), market_open=True)
        state.increment_error()


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler instance (created once, auto-started on app startup)
# ─────────────────────────────────────────────────────────────────────────────
_scheduler = BackgroundScheduler(timezone="UTC")
# Fire at :00, :05, :10 … seconds=10 gives 10-sec buffer for data availability
_scheduler.add_job(
    _run_cycle,
    trigger=CronTrigger(minute="*/5", second=10, timezone="UTC"),
    id="live_prediction_job",
    max_instances=1,
    coalesce=True,
    misfire_grace_time=60,
)


def start_scheduler():
    if not _scheduler.running:
        # Cold-start seeds (idempotent): running z-score stats from the training
        # gdelt_1h parquet, and a 24h LSTM window from the training latents so the
        # first cycle isn't zero-padded.
        sentiment.seed_news_stats()
        sentiment.seed_news_hourly(
            datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        )
        _scheduler.start()
        state.set_running(True)
        # Seed model_status immediately so the UI isn't stale until the first cycle.
        state.set_model_status("active" if check_data_source() else "inactive")
        logger.info("Started.")


def get_next_run_seconds() -> float | None:
    job = _scheduler.get_job("live_prediction_job")
    next_run = getattr(job, "next_run_time", None) if job else None
    if next_run:
        delta = (next_run - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    return None


def shutdown_scheduler():
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
