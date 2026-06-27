"""
train_tft.py
============
Train a Temporal Fusion Transformer (TFT) on FX pip-range prediction.

Requires Phase 3 + Phase 4 to be run first:
    python run_pipeline.py --only-phases 3 4 --pairs EURUSD AUDUSD USDJPY

Differences from model5/train_tft.py:
    - DATA_DIR     : pipeline_output/tft_data/  (Phase 4 output)
    - Target       : target_range (pip-scale, single model)  ← Phase 4
    - Quantiles    : [0.10, 0.50, 0.90]
    - Feature input: includes 16 LSTM latent cols + 4 sentiment scalars (Phase 3)
    - RobustScaler : already applied by Phase 4 (no re-scaling needed here)

Architecture:
    TemporalFusionTransformer (pytorch-forecasting)
    Encoder length    : 144 candles (12H at 5M)
    Prediction length : 1 candle  (5M ahead)
    Loss              : QuantileLoss([0.10, 0.50, 0.90])
    VSN               : auto-selects from 87 features

Usage:
    source /home/elio/my_workspace/envs/ravuslawenv/bin/activate
    python train_tft.py
    python train_tft.py --epochs 100 --batch-size 512 --hidden-size 128
    python train_tft.py --backtest-only
    python train_tft.py --resume-ckpt tft_models/tft_range-v1.ckpt
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

torch.serialization.add_safe_globals([GroupNormalizer, MultiNormalizer, QuantileLoss, MultiLoss, TimeSeriesDataSet])
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "pipeline_output" / "tft_data"   # Phase 4 output
MODELS_DIR  = BASE_DIR / "tft_models"
LOG_DIR     = BASE_DIR / "tft_logs"
RESULTS_CSV = BASE_DIR / "tft_backtest_results.csv"

MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Maximize Tensor Core Utilization (Ampere / Hopper)
torch.set_float32_matmul_precision("high")
pl.seed_everything(42, workers=True)

# Target (multi-target — volatility & direction)
TARGET      = ["target_upper_pip", "target_lower_pip"]
QUANTILES   = [0.10, 0.50, 0.90]   # Q10=lower bound, Q90=upper bound


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cfg_path = DATA_DIR / "feature_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            "feature_config.json not found!\n"
            "Run: python run_pipeline.py --only-phases 4 --pairs EURUSD AUDUSD USDJPY"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val   = pd.read_parquet(DATA_DIR / "val.parquet")
    test  = pd.read_parquet(DATA_DIR / "test.parquet")

    # Defensive cleanup
    for df in [train, val, test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0.0, inplace=True)

    # Clip any residual target outliers
    for t_col in TARGET:
        p001 = train[t_col].quantile(0.001)
        p999 = train[t_col].quantile(0.999)
        for df in [train, val, test]:
            df[t_col] = df[t_col].clip(p001, p999)

    # Pre-sort — makes TimeSeriesDataSet index building O(n)
    for df in [train, val, test]:
        df.sort_values(["group_id", "time_idx"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    log.info(f"  Loaded  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    log.info(f"  Groups  : {train['group_id'].unique().tolist()}")
    log.info(f"  Known   : {len(cfg['time_varying_known_reals'])} features")
    log.info(f"  Unknown : {len(cfg['time_varying_unknown_reals'])} features")
    return train, val, test, cfg


# ─────────────────────────────────────────────────────────────────────────────
# TimeSeriesDataSet Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(
    df: pd.DataFrame,
    cfg: dict,
    reference_dataset: TimeSeriesDataSet | None = None,
    predict: bool = False,
) -> TimeSeriesDataSet:
    """
    Wrap DataFrame as pytorch-forecasting TimeSeriesDataSet.
    If reference_dataset provided (val/test), build from reference
    to ensure consistent encoders/normalizers.
    """
    if reference_dataset is not None:
        return TimeSeriesDataSet.from_dataset(
            reference_dataset, df, predict=predict, stop_randomization=True
        )

    enc_len  = cfg["max_encoder_length"]
    pred_len = cfg["max_prediction_length"]

    known   = [c for c in cfg["time_varying_known_reals"]   if c not in TARGET]
    unknown = [c for c in cfg["time_varying_unknown_reals"] if c not in TARGET and c in df.columns]

    return TimeSeriesDataSet(
        df,
        time_idx                   = "time_idx",
        target                     = TARGET,
        group_ids                  = ["group_id"],
        min_encoder_length         = enc_len // 2,
        max_encoder_length         = enc_len,
        min_prediction_length      = pred_len,
        max_prediction_length      = pred_len,
        static_categoricals        = ["group_id"],
        time_varying_known_reals   = known,
        time_varying_unknown_reals = unknown,
        target_normalizer          = MultiNormalizer([
            GroupNormalizer(groups=["group_id"], center=True, scale_by_group=True),
            GroupNormalizer(groups=["group_id"], center=True, scale_by_group=True)
        ]),
        add_relative_time_idx      = True,
        add_target_scales          = True,
        add_encoder_length         = True,
        allow_missing_timesteps    = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class RangeTFT(TemporalFusionTransformer):
    """TFT with bfloat16 TensorBoard crash fix."""
    def log_prediction(self, *args, **kwargs):
        if self.trainer.precision in ["bf16", "bf16-mixed"]:
            return
        return super().log_prediction(*args, **kwargs)


def build_tft(
    dataset: TimeSeriesDataSet,
    hidden_size: int = 128,
    attention_heads: int = 4,
    dropout: float = 0.15,
    hidden_cont_size: int = 32,
    lr: float = 3e-4,
) -> RangeTFT:
    return RangeTFT.from_dataset(
        dataset,
        learning_rate          = lr,
        hidden_size            = hidden_size,
        attention_head_size    = attention_heads,
        dropout                = dropout,
        hidden_continuous_size = hidden_cont_size,
        loss                   = MultiLoss([QuantileLoss(quantiles=QUANTILES), QuantileLoss(quantiles=QUANTILES)]),
        log_interval           = 50,
        reduce_on_plateau_patience = 4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    cfg:         dict,
    epochs:      int   = 100,
    batch_size:  int   = 512,
    hidden_size: int   = 128,
    attention_heads: int = 4,
    dropout:     float = 0.15,
    lr:          float = 3e-4,
    patience:    int   = 10,
    num_workers: int   = 8,
    precision:   str   = "bf16-mixed",
    accum_grad:  int   = 2,
    resume_ckpt: str | None = None,
) -> RangeTFT:

    train_ds = build_dataset(train_df, cfg)
    val_ds   = build_dataset(val_df,   cfg, reference_dataset=train_ds)

    train_loader = train_ds.to_dataloader(
        train=True,  batch_size=batch_size,     num_workers=num_workers,
        shuffle=True, persistent_workers=True,   pin_memory=True,
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=num_workers,
        persistent_workers=True, pin_memory=True,
    )

    tft = build_tft(train_ds, hidden_size, attention_heads, dropout,
                    hidden_size // 4, lr)
    n_params = sum(p.numel() for p in tft.parameters())
    log.info(f"  TFT parameters : {n_params:,}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        ModelCheckpoint(
            monitor="val_loss", dirpath=str(MODELS_DIR),
            filename="tft_range", save_top_k=1, mode="min", save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs              = epochs,
        accelerator             = "gpu" if torch.cuda.is_available() else "cpu",
        devices                 = 1,
        precision               = precision,
        gradient_clip_val       = 0.5,
        accumulate_grad_batches = accum_grad,
        callbacks               = callbacks,
        logger                  = TensorBoardLogger(str(LOG_DIR), name="range"),
        enable_progress_bar     = True,
        log_every_n_steps       = 100,
        check_val_every_n_epoch = 2,
    )

    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt,
    )

    best_ckpt = trainer.checkpoint_callback.best_model_path
    log.info(f"  Best checkpoint : {best_ckpt}")
    return TemporalFusionTransformer.load_from_checkpoint(best_ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# OOS Backtest
# ─────────────────────────────────────────────────────────────────────────────
def backtest_oos(
    model:    TemporalFusionTransformer,
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    cfg:      dict,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Evaluate OOS coverage: did actual pip-range fall inside [Q10, Q90]?
    """
    enc_len    = cfg["max_encoder_length"]
    train_tail = train_df.groupby("group_id", group_keys=False).tail(enc_len)
    test_ctx   = pd.concat([train_tail, test_df]).sort_values(["group_id", "time_idx"])

    train_ds = build_dataset(train_df, cfg)
    test_ds  = build_dataset(test_ctx, cfg, reference_dataset=train_ds, predict=False)
    loader   = test_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=4)

    log.info(f"\n  Running OOS Backtest (test rows={len(test_df):,}) ...")
    preds = model.predict(loader, return_y=True, mode="quantiles")

    # preds.output shape for MultiLoss: tuple of 2 elements, each (n, 1, n_quantiles=3)
    out_upper = preds.output[0].cpu().numpy()[:, 0, :]
    out_lower = preds.output[1].cpu().numpy()[:, 0, :]

    # preds.y is a tuple (targets, None) -> targets is a tuple of 2 tensors
    y_upper = preds.y[0][0].cpu().numpy().flatten() if preds.y is not None else np.zeros_like(out_upper[:, 1])
    y_lower = preds.y[0][1].cpu().numpy().flatten() if preds.y is not None else np.zeros_like(out_lower[:, 1])

    # Extract time indices to map back to DateTime
    df_preds = pd.DataFrame({
        "group_id": test_ds.decoded_index["group_id"],
        "time_idx": test_ds.decoded_index["time_idx_first_prediction"],
        "Actual_Upper": y_upper,
        "Actual_Lower": y_lower,
        "Upper_Q10": out_upper[:, 0],
        "Upper_Q50": out_upper[:, 1],
        "Upper_Q90": out_upper[:, 2],
        "Lower_Q10": out_lower[:, 0],
        "Lower_Q50": out_lower[:, 1],
        "Lower_Q90": out_lower[:, 2],
    })

    # Keep only predictions for actual test_df rows (removes imputed gaps & train_tail overlap)
    test_cols = ["group_id", "time_idx", "DateTime_UTC"] if "DateTime_UTC" in test_df.columns else ["group_id", "time_idx"]
    df_preds = df_preds.merge(test_df[test_cols], on=["group_id", "time_idx"], how="inner")

    preds_path = DATA_DIR / "tft_predictions.csv"
    df_preds.to_csv(preds_path, index=False)
    log.info(f"  ✓ Saved row-by-row predictions to {preds_path}")

    # Simplified summary results
    results = pd.DataFrame([{
        "model"        : "TFT-Multi-Target",
        "quantiles"    : str(QUANTILES),
        "n_samples"    : len(y_upper),
    }])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train TFT for FX pip-range prediction")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--hidden-size",  type=int,   default=32)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--dropout",      type=float, default=0.15)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--workers",      type=int,   default=8)
    parser.add_argument("--precision",    type=str,   default="bf16-mixed",
                        choices=["bf16-mixed", "16-mixed", "32-true"])
    parser.add_argument("--accum-grad",   type=int,   default=4,
                        help="Gradient accumulation batches")
    parser.add_argument("--backtest-only", action="store_true",
                        help="Skip training — run backtest from saved checkpoint")
    parser.add_argument("--resume-ckpt",  type=str,   default=None,
                        metavar="PATH",   help="Resume training from checkpoint")
    parser.add_argument("--ckpt",         type=str,   default=None,
                        metavar="PATH",   help="Checkpoint to load for --backtest-only")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    log.info("=" * 65)
    log.info("  TFT Training — FX Pip-Range Prediction (Q10/Q50/Q90)")
    log.info("=" * 65)
    log.info(f"  Target      : {TARGET}")
    log.info(f"  Quantiles   : {QUANTILES}")
    log.info(f"  Epochs      : {args.epochs}  (patience={args.patience})")
    log.info(f"  Batch size  : {args.batch_size}")
    log.info(f"  Hidden size : {args.hidden_size}")
    log.info(f"  LR          : {args.lr}")
    log.info(f"  Precision   : {args.precision}")
    log.info(f"  GPU         : {torch.cuda.is_available()}")

    train_df, val_df, test_df, cfg = load_data()

    # ── Backtest-only mode ────────────────────────────────────────────────
    if args.backtest_only:
        ckpt = args.ckpt or str(MODELS_DIR / "tft_range.ckpt")
        if not Path(ckpt).exists():
            log.error(f"Checkpoint not found: {ckpt}")
            log.error("Run training first or specify --ckpt PATH")
            return
        log.info(f"  Loading checkpoint: {ckpt}")
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt)
        results = backtest_oos(model, train_df, test_df, cfg, args.batch_size)
        results.to_csv(RESULTS_CSV, index=False)
        log.info(f"\n  Results saved → {RESULTS_CSV}")
        log.info("\n" + results.to_string(index=False))
        return

    # ── Training ─────────────────────────────────────────────────────────
    model = train_model(
        train_df, val_df, cfg,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        hidden_size  = args.hidden_size,
        attention_heads = args.attention_heads,
        dropout      = args.dropout,
        lr           = args.lr,
        patience     = args.patience,
        num_workers  = args.workers,
        precision    = args.precision,
        accum_grad   = args.accum_grad,
        resume_ckpt  = args.resume_ckpt,
    )

    # ── OOS Backtest ─────────────────────────────────────────────────────
    results = backtest_oos(model, train_df, test_df, cfg, args.batch_size)
    results.to_csv(RESULTS_CSV, index=False)
    log.info(f"\n  Results saved → {RESULTS_CSV}")
    log.info("\n" + results.to_string(index=False))
    log.info("\n  Done. ✓")


if __name__ == "__main__":
    main()

