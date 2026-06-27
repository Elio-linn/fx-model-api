"""
api/predictor/features.py
Feature engineering and TFT prediction logic.
Ported from live_backtest_tft.py sections 4 & 5.
"""
import numpy as np
import pandas as pd
import torch

from api.config import CURRENCIES, PAIR, TIMEFRAME, DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# Technical Analysis Features
# ─────────────────────────────────────────────────────────────────────────────
def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df["return_5M"] = df["Close"].pct_change() * 10000
    df["body_ratio_5M"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-8)

    # ATR (14)
    tr1 = df["High"] - df["Low"]
    tr2 = abs(df["High"] - df["Close"].shift())
    tr3 = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df["atr_ratio_5M"] = tr / (atr + 1e-8)
    df["atr"] = atr

    # EMA 20
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    df["ema_slope_20_5M"] = ema20.diff()

    # Bollinger Width
    std20 = df["Close"].rolling(20).std()
    df["bollinger_width_20_5M"] = (4 * std20) / ema20

    # RSI 14
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14_5M"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist_5M"] = macd - signal

    # Hourly placeholders (not available in pure live mode)
    df["distance_to_ema_1h"] = 0.0
    df["ema_hourly_state"] = 0.0
    df["ema_stack_state_5M"] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Hourly LSTM Sentiment Encoding
# ─────────────────────────────────────────────────────────────────────────────
# How many trailing hours the sentiment encoder consumes — matches the
# seq_len=24 sliding window used in phase2_lstm_encoder.py training.
LSTM_WINDOW_HOURS = 24


def compute_latents(
    news_windows: dict[str, np.ndarray],
    lstm_encoders: dict,
) -> dict[str, np.ndarray]:
    """
    Encode each currency's 24-hour sentiment window into a latent vector.

    `news_windows` maps currency → array of shape (LSTM_WINDOW_HOURS, 2),
    each row [combined_z, ma_z] for one past hour, oldest-first — a real
    sliding window built from stored hourly news (not a repeated value).

    Runs once per hour; the result is cached by the scheduler and reused on
    every 5-minute TFT cycle within the hour.
    """
    latents = {}
    for cur in CURRENCIES:
        window = np.asarray(news_windows[cur], dtype=np.float32)
        seq = torch.tensor(window[np.newaxis, ...], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            latents[cur] = lstm_encoders[cur].encode(seq).cpu().numpy()[0]
    return latents


# ─────────────────────────────────────────────────────────────────────────────
# Live Prediction Input Builder
# ─────────────────────────────────────────────────────────────────────────────
def run_live_prediction(
    history_df: pd.DataFrame,
    latents: dict,
    feature_config: dict,
    ff_aggregates: dict | None = None,
) -> pd.DataFrame | None:
    """
    Build a TFT-compatible DataFrame for the latest encoder window.

    `latents` maps currency → encoded sentiment vector, precomputed once per
    hour by compute_latents(). The TA features are still computed every 5-min
    cycle; the news latents are forward-filled across the hour, exactly as the
    1H news was merge_asof'd into 5M candles during training.

    `ff_aggregates` maps "base"/"quote" → {event_count, has_high_impact} for the
    trailing hour. Only these two bounded, cleanly-scaled FF columns are filled;
    Impact_Score_sum / Surprise_Factor_sum stay zero because the training data
    carried unit artifacts (|values| up to 1e11), so the dataset normalizer would
    crush any realistic live surprise to ≈0 anyway — populating them only risks a
    scale-mismatch spike.

    Returns None if insufficient history.
    """
    df = compute_ta_features(history_df.copy())

    # Inject latents into DataFrame
    for i in range(8):
        df[f"base_latent_{i}"] = latents["EUR"][i]
        df[f"base_latent_{i}_x_atr"] = df[f"base_latent_{i}"] * df["atr"]
        df[f"quote_latent_{i}"] = latents["USD"][i]
        df[f"quote_latent_{i}_x_atr"] = df[f"quote_latent_{i}"] * df["atr"]

    # FF aggregate columns: fill the clean bounded pair from this hour's events;
    # keep the artifact-scaled magnitude pair zeroed (see docstring).
    ff_aggregates = ff_aggregates or {}
    for prefix in ["base", "quote"]:
        agg = ff_aggregates.get(prefix, {})
        df[f"{prefix}_Impact_Score_sum"] = 0.0
        df[f"{prefix}_Surprise_Factor_sum"] = 0.0
        df[f"{prefix}_event_count"] = float(agg.get("event_count", 0))
        df[f"{prefix}_has_high_impact"] = float(agg.get("has_high_impact", 0))

    df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    enc_len = feature_config["max_encoder_length"]
    if len(df) < enc_len:
        return None

    pred_df = df.tail(enc_len).copy()
    pred_df["group_id"] = f"{PAIR}_{TIMEFRAME}"
    pred_df["time_idx"] = np.arange(enc_len)
    pred_df["target_upper_pip"] = 0.0
    pred_df["target_lower_pip"] = 0.0
    pred_df["pair"] = PAIR
    pred_df["timeframe"] = TIMEFRAME

    # Ensure all required TFT columns exist
    missing_cols = set(feature_config["time_varying_unknown_reals"]) - set(pred_df.columns)
    for c in missing_cols:
        pred_df[c] = 0.0

    return pred_df
