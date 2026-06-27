"""
api/config.py
Central configuration for the Live TFT Backtest API.
All paths and settings are sourced from environment variables or defaults.
"""
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load api/.env regardless of the current working directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(os.getenv("BASE_DIR", "/mnt/data/fx-model-api/api"))
PIPELINE_OUT = BASE_DIR / "pipeline_output"
LSTM_WEIGHTS_DIR = PIPELINE_OUT / "lstm_weights"
LSTM_LATENTS_DIR = PIPELINE_OUT / "lstm_latents"   # training latents (combined_z/ma_z seed)
GDELT_1H_DIR = PIPELINE_OUT / "gdelt_1h"           # training WM/MA (z-score seed)
TFT_MODELS_DIR = BASE_DIR / "tft_models"
TFT_DATA_DIR = PIPELINE_OUT / "tft_data"

# ─────────────────────────────────────────────────────────────────────────────
# Model / Trading Config
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENCIES = ["EUR", "USD"]
PAIR = "EURUSD"
TIMEFRAME = "5M"
OHLCV_COUNT = 500          # candles to fetch per cycle
MODEL_VERSION = os.getenv("MODEL_VERSION", "hybrid_model_4")  # shown in /status


def pip_divisor(pair: str = PAIR) -> float:
    """
    Price units per pip. JPY-quoted pairs use 0.01 (÷100); all others 0.0001 (÷10000).
    pip ↔ price:  pip = price_diff * divisor ;  price = pip / divisor
    """
    return 100.0 if pair.upper().endswith("JPY") else 10000.0

# External OHLC data source (broker feed). Override with env var if needed.
OHLCV_API_BASE = os.getenv("OHLCV_API_BASE", "http://18.138.136.233:8000")

# ─────────────────────────────────────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "fx_model")

# ─────────────────────────────────────────────────────────────────────────────
# API Config
# ─────────────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Live TFT Backtest API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "FastAPI wrapper for the Live Temporal Fusion Transformer "
    "backtest engine. Runs a 5-minute scheduler, stores predictions "
    "in CSV, and exposes REST endpoints for monitoring."
)
