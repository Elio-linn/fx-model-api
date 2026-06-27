# Live TFT Backtest API

This is a FastAPI wrapper for the Temporal Fusion Transformer (TFT) Live Backtesting Engine. It runs a 5-minute background scheduler to predict forex price ranges (EURUSD) using a pre-trained TFT model and LSTM sentiment encoders, and stores the predictions safely in a local CSV datastore.

## 🏗️ Architecture
- **Framework**: FastAPI
- **Background Jobs**: APScheduler (CronTrigger aligned to 5-minute intervals)
- **Data Storage**: Thread-safe CSV append (`api/data/predictions_YYYYMMDD.csv`)
- **Data Sources**:
  - OHLCV: `yfinance`
  - Sentiment: `GDELT Project`
  - Macro Events: `Forex Factory`
- **Models**:
  - PyTorch Forecasting (TFT)
  - PyTorch (LSTM Encoders)

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ and a virtual environment set up.

```bash
# Activate your virtual environment
source /home/elio/my_workspace/envs/ravuslawenv/bin/activate

# Install required packages
pip install fastapi "uvicorn[standard]" apscheduler python-dotenv
```
*(Note: It is assumed that `torch`, `pytorch-forecasting`, `pandas`, and `yfinance` are already installed in your environment).*

### 2. Environment Configuration
Copy the `.env` template if needed, or rely on defaults.
```bash
cd /mnt/data/hybridmodel4/api
cp .env.example .env  # (If template exists)
```
**Default `.env` settings:**
- `BASE_DIR`: `/mnt/data/hybridmodel4`
- `API_PORT`: `8000`

---

## 🏃‍♂️ Running the Server

Start the API server using Uvicorn. **Always run this from the project root directory** so that Python can correctly resolve relative imports to the TFT codebase.

```bash
cd /mnt/data/hybridmodel4
source /home/elio/my_workspace/envs/ravuslawenv/bin/activate

# Start the server (with auto-reload for development)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

When the server starts, it will:
1. Load the PyTorch models into memory (may take a few seconds).
2. Start the 5-minute background scheduler.
3. Be ready to accept HTTP requests.

---

## 🔌 API Endpoints

Once running, interactive API documentation is automatically available at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### `GET /status`
Returns the current status of the bot, including whether the scheduler is running, the time of the last evaluated candle, countdown to the next run, and a summary of coverage accuracy.

### `POST /control/start`
Starts or resumes the background 5-minute prediction scheduler.

### `POST /control/stop`
Pauses the background scheduler. (Existing predictions remain safe).

### `GET /predictions`
Returns a paginated list of all predictions.
- Query Params: `page` (default: 1), `page_size` (default: 20), `date_from`, `date_to`

### `GET /predictions/latest`
Returns the single most recent prediction row.

### `GET /predictions/accuracy`
Returns the live calculated accuracy (Hit Rate) based on predictions that have completed their 5-minute lifecycle.

### `GET /predictions/export.csv`
Downloads the entire prediction history as a single `.csv` file.

---

## 📁 Folder Structure

```text
api/
├── __init__.py
├── .env                  # Environment variables override
├── config.py             # Central configuration & paths
├── main.py               # FastAPI entry point & lifespan events
├── schemas.py            # Pydantic models for API responses
├── scheduler.py          # APScheduler background job logic
├── state.py              # Thread-safe in-memory bot state
├── storage.py            # Thread-safe CSV reading and writing
├── data/                 # Auto-generated directory for CSV files
│   └── predictions_YYYYMMDD.csv
├── predictor/            # Core logic ported from live_backtest_tft.py
│   ├── __init__.py
│   ├── features.py       # Technical analysis & TFT input builder
│   ├── fetchers.py       # yfinance, GDELT, and Forex Factory fetchers
│   └── model_loader.py   # Singleton loader for PyTorch models
└── routers/              # FastAPI endpoint routers
    ├── __init__.py
    ├── control.py
    ├── predictions.py
    └── status.py
```

---

## 💡 Notes on Accuracy Calculation (0.0 Values)
During low volatility periods (e.g., Asian Session), the 5-minute OHLCV data from `yfinance` may report identical values for Open, High, Low, and Close. When this happens, the actual pip movement is mathematically `0.0`, resulting in `upper=0.0` and `lower=0.0` in the terminal logs. This is correct behavior representing a flat market.
