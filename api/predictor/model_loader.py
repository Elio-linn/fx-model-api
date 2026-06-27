"""
api/predictor/model_loader.py
Loads and caches TFT and LSTM encoder models at startup.
Models are loaded once and reused across all scheduler cycles.
"""
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer
import json

from api.config import (
    DEVICE, LSTM_WEIGHTS_DIR, TFT_MODELS_DIR, TFT_DATA_DIR,
    CURRENCIES
)

# ─────────────────────────────────────────────────────────────────────────────
# LSTM Encoder (matches phase2_lstm_encoder.py architecture)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMSentimentEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, latent_dim=16, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.inter_dropout = nn.Dropout(p=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.proj = nn.Linear(hidden_size, latent_dim)

    def encode(self, x):
        out1, _ = self.lstm1(x)
        out1_d = self.inter_dropout(out1)
        out2, _ = self.lstm2(out1_d)
        return self.proj(out2[:, -1, :])


# ─────────────────────────────────────────────────────────────────────────────
# Singleton holders (populated at startup via load_models())
# ─────────────────────────────────────────────────────────────────────────────
_tft_model = None
_lstm_encoders = {}
_feature_config = {}
_ref_dataset = None


def load_models():
    """
    Called once at FastAPI startup (lifespan).
    Loads TFT checkpoint, LSTM weights, feature config, and reference dataset.
    Raises RuntimeError if any required file is missing.
    """
    global _tft_model, _lstm_encoders, _feature_config, _ref_dataset
    import pandas as pd
    import sys

    # Ensure hybridmodel4 root is in path for train_tft import
    import os
    root = str(TFT_DATA_DIR.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    # 1. Feature config
    cfg_path = TFT_DATA_DIR / "feature_config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"feature_config.json not found at {cfg_path}")
    with open(cfg_path) as f:
        _feature_config = json.load(f)

    # 2. TFT model
    ckpt_path = TFT_MODELS_DIR / "tft_range.ckpt"
    if not ckpt_path.exists():
        raise RuntimeError(f"TFT checkpoint not found at {ckpt_path}")
    _tft_model = TemporalFusionTransformer.load_from_checkpoint(
        str(ckpt_path), map_location=DEVICE
    )
    _tft_model.eval()

    # 3. LSTM encoders
    for cur in CURRENCIES:
        weights_path = LSTM_WEIGHTS_DIR / f"lstm_encoder_{cur}.pt"
        if not weights_path.exists():
            raise RuntimeError(f"LSTM weights not found at {weights_path}")
        enc = LSTMSentimentEncoder(latent_dim=16).to(DEVICE)
        enc.load_state_dict(
            torch.load(str(weights_path), map_location=DEVICE), strict=False
        )
        enc.eval()
        _lstm_encoders[cur] = enc

    # 4. Reference dataset
    train_parquet = TFT_DATA_DIR / "train.parquet"
    if not train_parquet.exists():
        raise RuntimeError(f"train.parquet not found at {train_parquet}")
    from train_tft import build_dataset
    train_ref = pd.read_parquet(str(train_parquet))
    _ref_dataset = build_dataset(train_ref, _feature_config)

    print(f"[model_loader] All models loaded successfully on {DEVICE}.")


def get_tft() -> TemporalFusionTransformer:
    if _tft_model is None:
        raise RuntimeError("TFT model not loaded. Call load_models() first.")
    return _tft_model


def get_lstm_encoders() -> dict:
    if not _lstm_encoders:
        raise RuntimeError("LSTM encoders not loaded. Call load_models() first.")
    return _lstm_encoders


def get_feature_config() -> dict:
    return _feature_config


def get_ref_dataset():
    return _ref_dataset
