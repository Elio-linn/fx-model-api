# FX Model API

Live forex range-prediction service for **EURUSD (5-minute)**. A background scheduler
fetches market + sentiment data every 5 minutes, runs a **Temporal Fusion Transformer (TFT)**
with **LSTM sentiment encoders**, stores everything in **MongoDB**, and exposes a small REST API.

---

## 🏗️ Architecture

```
                 ┌──────────────── every 5 min (APScheduler, UTC-aligned) ────────────────┐
                 │                                                                          │
  External OHLC API ──┴──────────────────────► fetchers ──► TFT predict ─┐                  ▼
                                                              ▲          ├─► MongoDB ──► REST API
  GDELT 2.0 events ───┐                                       │ latents  │   ohlcv / gdelt
  Forex Factory   ────┴──► hourly news → LSTM encode (24h) ───┘          │   gdelt_raw / ff_events
                          └──────── once per hour, cached ───────┘          news_hourly / predictions
```

- **Framework**: FastAPI + Uvicorn
- **Scheduler**: APScheduler `CronTrigger("*/5", second=10)` — auto-starts on boot, skips weekends
- **Storage**: MongoDB (natural-key upsert → automatic dedup across cycles)
- **Model**: PyTorch-Forecasting TFT + PyTorch LSTM encoders (GPU if available)
- **News cadence (1h variant)**: GDELT + Forex Factory + the LSTM sentiment
  encoding refresh **once per hour**, not every cycle. The model was trained on
  1H news merge_asof'd into 5M candles, so the news latents only change hourly.
  The TFT itself still predicts every 5 min, reusing the hour's cached latents;
  the encoder is fed a real 24-hour `[combined_z, ma_z]` window assembled from
  the `news_hourly` collection.
- **Training-parity news features**:
  - *GDELT*: the trailing hour (4×15-min batches) is aggregated into
    `Weighted_Momentum` / `Market_Attention` (phase1 formula), then turned into
    `ws_z`/`ma_z` by a streaming **expanding z-score** whose running stats are
    seeded once from `pipeline_output/gdelt_1h/*.parquet` (so hour 1 already
    z-scores against years of history). Stats live in the `news_stats` collection.
  - *Cold start*: on first boot the 24h LSTM window is seeded from the training
    `lstm_latents/*.parquet` (its `combined_z`/`ma_z`), so the first cycle isn't
    zero-padded; seeded hours age out as live hours arrive.
  - *Forex Factory*: `event_count` and `has_high_impact` (clean, bounded) are
    populated per hour into the `base_/quote_` TFT columns. `Impact_Score_sum` /
    `Surprise_Factor_sum` are left zero — the training data carried unit
    artifacts (|values| up to 1e11) that would crush realistic live values to ≈0.

### Data sources
| Source | What | Cadence |
|--------|------|---------|
| OHLC API (`OHLCV_API_BASE`) | 5-min candles | every cycle (5 min) |
| GDELT 2.0 events | sentiment per currency (actor-code matched) + raw events | once per hour |
| Forex Factory | this-week economic calendar (High/Medium) | once per hour |
| LSTM sentiment encoding | 24h `[combined_z, ma_z]` window → per-currency latents | once per hour (cached) |

---

## 📋 Prerequisites

- **Python 3.10+**
- **MongoDB** running and reachable (default `mongodb://localhost:27017`)
- **CUDA GPU** (optional — falls back to CPU)
- Trained model files under `BASE_DIR` (see below)

Required model files (relative to `BASE_DIR`):
```
tft_models/tft_range.ckpt
pipeline_output/tft_data/feature_config.json
pipeline_output/tft_data/train.parquet
pipeline_output/lstm_weights/lstm_encoder_EUR.pt
pipeline_output/lstm_weights/lstm_encoder_USD.pt
```

---

## ⚙️ Installation

```bash
cd /mnt/data/fx-model-api/api
python3 -m venv .venv && source .venv/bin/activate   # or use an existing venv
pip install -r requirements.txt
```

---

## 🔧 Configuration

All settings come from environment variables (loaded from `api/.env`). Defaults in `api/config.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | `/mnt/data/fx-model-api/api` | Root holding the model files |
| `OHLCV_API_BASE` | `http://18.138.136.233:8000` | External OHLC candle API |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `fx_model` | Database name |
| `MODEL_VERSION` | `hybrid_model_4` | Version label shown in `/status` |
| `API_HOST` | `0.0.0.0` | Bind host |
| `API_PORT` | `8000` | Bind port |

Example `api/.env`:
```env
BASE_DIR=/mnt/data/fx-model-api/api
OHLCV_API_BASE=http://18.138.136.233:8000
MONGO_URI=mongodb://localhost:27017
MONGO_DB=fx_model
MODEL_VERSION=hybrid_model_4
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 🏃 Running locally

> **Run from `/mnt/data/fx-model-api/api`** so the `api.main:app` import resolves.

```bash
cd /mnt/data/fx-model-api/api
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

On startup it: ensures MongoDB indexes → loads models (~5s on GPU) → starts the 5-min scheduler.

Swagger UI → **http://localhost:8000/docs**

> ⚠️ **Single worker only.** The scheduler is a singleton; do **not** run multiple
> Uvicorn/Gunicorn workers or the prediction cycle will run more than once.

---

## 🔌 API Endpoints

| Method | Path | Params | Description |
|--------|------|--------|-------------|
| GET | `/` | — | Health check |
| GET | `/status` | — | Bot state, market status, `model_status`, `model_version`, accuracy |
| GET | `/ohlcv` | `pair`, `limit` | Recent candles (newest first) |
| GET | `/predictions` | `pair`, `limit` | Recent predictions (newest first) |
| GET | `/predictions/latest` | — | Most recent prediction |
| GET | `/predictions/accuracy` | — | Coverage accuracy stats |

`pair` defaults to `EURUSD`, `limit` defaults to `100` (max 1000).

```bash
curl "http://localhost:8000/status"
curl "http://localhost:8000/ohlcv?pair=EURUSD&limit=10"
curl "http://localhost:8000/predictions?pair=EURUSD&limit=10"
```

---

## 🗄️ MongoDB collections (`fx_model`)

| Collection | Unique key | Contents |
|------------|-----------|----------|
| `ohlcv` | `(pair, time)` | 5-min OHLCV candles |
| `gdelt` | `(batch_time, currency)` | Aggregated sentiment (`ws_z`, `ma_z`) |
| `gdelt_raw` | `global_event_id` | Currency-relevant raw GDELT events |
| `ff_events` | `(event_time, country, title)` | Economic calendar events |
| `news_hourly` | `(hour, currency)` | 1H GDELT+FF sentiment (`combined_z`, `ma_z`) feeding the LSTM window |
| `news_stats` | `(currency)` | Running WM/MA stats (`n`, `mean`, `M2`) for the expanding z-score |
| `predictions` | `(pair, target_time)` + `id` | Model output (pip + price quantiles, actuals) |
| `counters` | `_id` | Auto-increment id sequence |

Indexes are created automatically on startup. Wipe a stale DB with:
```bash
mongosh fx_model --eval "db.dropDatabase()"
```

---

## 🚀 Deploy on a server (systemd)

1. **Install MongoDB** and ensure it is running:
   ```bash
   sudo systemctl enable --now mongod
   ```

2. **Place the code + model files** on the server and set `BASE_DIR` accordingly in `api/.env`.

3. **Install deps** into a venv (as above).

4. **Create a systemd unit** `/etc/systemd/system/fx-model-api.service`:
   ```ini
   [Unit]
   Description=FX Model API
   After=network.target mongod.service
   Requires=mongod.service

   [Service]
   Type=simple
   User=elio
   WorkingDirectory=/mnt/data/fx-model-api/api
   ExecStart=/home/elio/my_workspace/envs/ravuslawenv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
   Restart=always
   RestartSec=5
   # GPU access if used:
   # Environment=CUDA_VISIBLE_DEVICES=0

   [Install]
   WantedBy=multi-user.target
   ```

5. **Enable & start**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now fx-model-api
   sudo systemctl status fx-model-api
   journalctl -u fx-model-api -f          # follow logs
   ```

### Optional: nginx reverse proxy
```nginx
server {
    listen 80;
    server_name your.domain;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

> CORS is currently open (`allow_origins=["*"]` in `api/main.py`). Restrict it for production.

---

## 🩺 Operations

- **Logs**: structured via Python `logging` (stdout / `journalctl`). Key lines:
  - `[scheduler] Predicted for HH:MM → …` — successful cycle
  - `candle missing: …` — data source problem
  - `Market closed (weekend). Skipping cycle.` — expected on weekends
- **Health**: `GET /status` → `running`, `model_status` (data API reachable), `market_open`.
- **Deploying a new model**: replace the `.ckpt`, bump `MODEL_VERSION` in `api/.env`, restart the service.

---

## 📁 Layout

```
api/
├── requirements.txt
├── api/
│   ├── main.py            # FastAPI app + lifespan (startup/shutdown)
│   ├── config.py          # env-driven settings + pip_divisor()
│   ├── db.py              # MongoDB client, collections, indexes
│   ├── state.py           # in-memory bot state
│   ├── storage.py         # MongoDB read/write for all collections
│   ├── scheduler.py       # 5-min cycle (fetch → predict → persist)
│   ├── schemas.py         # Pydantic response models
│   ├── .env               # environment overrides
│   ├── predictor/
│   │   ├── fetchers.py    # OHLCV / GDELT / Forex Factory
│   │   ├── features.py    # TFT input builder
│   │   └── model_loader.py
│   └── routers/
│       ├── status.py
│       ├── predictions.py
│       └── ohlcv.py
└── tft_models/ · pipeline_output/   # model artifacts (under BASE_DIR)
```
