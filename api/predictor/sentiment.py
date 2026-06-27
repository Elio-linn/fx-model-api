"""
api/predictor/sentiment.py
Streaming expanding-z-score for GDELT sentiment + cold-start seeders.

Training (phase1_gdelt_aggregator.py) normalized the hourly Weighted_Momentum
and Market_Attention with an anti-leakage Expanding_Z_Score(shift=1): at hour t
the z-score uses the mean/std of all hours before t. Live we reproduce that as a
streaming statistic — seeded once from the training parquet so the very first
live hour already z-scores against years of history — then Welford-updated each
hour. The running stats live in the `news_stats` collection.
"""
import math
import logging
from datetime import timedelta

import pandas as pd

from api.config import CURRENCIES, GDELT_1H_DIR, LSTM_LATENTS_DIR
from api import storage

logger = logging.getLogger(__name__)

_EMPTY = {"n": 0, "mean": 0.0, "M2": 0.0}


def _series_moments(series: pd.Series) -> dict:
    """Count / mean / sum-of-squared-deviations (M2) for a parquet series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(s.shape[0])
    if n == 0:
        return dict(_EMPTY)
    mean = float(s.mean())
    M2 = float(((s - mean) ** 2).sum())
    return {"n": n, "mean": mean, "M2": M2}


def zscore_update(stat: dict, x: float) -> tuple[float, dict]:
    """
    Expanding z-score of `x` against the running `stat`, then fold `x` in.

    Returns (z, new_stat). The z-score uses the stats over all *prior* points
    only (Welford update applied afterwards), matching phase1's shift(1).
    """
    n, mean, M2 = stat.get("n", 0), stat.get("mean", 0.0), stat.get("M2", 0.0)
    if n >= 2:
        std = math.sqrt(M2 / (n - 1))
        z = (x - mean) / std if std > 1e-8 else 0.0
    else:
        z = 0.0
    # Welford online update
    n2 = n + 1
    delta = x - mean
    mean2 = mean + delta / n2
    M2_2 = M2 + delta * (x - mean2)
    return z, {"n": n2, "mean": mean2, "M2": M2_2}


def seed_news_stats() -> None:
    """
    Seed the running WM/MA stats from the training gdelt_1h parquet — once per
    currency (skipped if stats already exist, so restarts keep accumulating).
    """
    for cur in CURRENCIES:
        if storage.get_news_stats(cur) is not None:
            continue
        path = GDELT_1H_DIR / f"gdelt_1h_{cur}.parquet"
        if not path.exists():
            logger.warning("seed_news_stats: %s missing — z-score starts cold", path)
            storage.save_news_stats(cur, {"wm": dict(_EMPTY), "ma": dict(_EMPTY)})
            continue
        df = pd.read_parquet(path)
        stats = {
            "wm": _series_moments(df["Weighted_Momentum"]),
            "ma": _series_moments(df["Market_Attention"]),
        }
        storage.save_news_stats(cur, stats)
        logger.info(
            "seed_news_stats %s: wm(n=%d mean=%.3f) ma(n=%d mean=%.1f)",
            cur, stats["wm"]["n"], stats["wm"]["mean"],
            stats["ma"]["n"], stats["ma"]["mean"],
        )


def seed_news_hourly(hour_now) -> None:
    """
    Cold-start the LSTM window: if a currency has no hourly news rows yet, seed
    the 24 hours preceding `hour_now` from the training lstm_latents parquet
    (its combined_z / ma_z). The first live cycle then has a full 24h window
    instead of a zero-padded one; the seeded rows age out as live hours arrive.
    """
    for cur in CURRENCIES:
        if storage.count_news_hourly(cur) > 0:
            continue
        path = LSTM_LATENTS_DIR / f"lstm_latents_{cur}.parquet"
        if not path.exists():
            logger.warning("seed_news_hourly: %s missing — window starts zero-padded", path)
            continue
        tail = pd.read_parquet(path)[["combined_z", "ma_z"]].tail(24)
        for k, (_, row) in enumerate(tail.iterrows()):
            hour = hour_now - timedelta(hours=24 - k)   # hours -24 .. -1
            storage.save_news_hourly(hour, {cur: {
                "combined_z": float(row["combined_z"]),
                "ma_z": float(row["ma_z"]),
                "ws_z": float(row["combined_z"]),   # split unknown — combined as proxy
                "sf_z": 0.0,
            }})
        logger.info("seed_news_hourly %s: seeded 24h window from training latents", cur)
