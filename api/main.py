"""
api/main.py
FastAPI application entry point.

Lifespan:
  startup  → load_models(), start scheduler (auto-runs on boot)
  shutdown → stop scheduler

Routers:
  /status       — bot state & accuracy summary
  /predictions  — list, latest, by id, accuracy, CSV export
"""
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Ensure project root on sys.path ──────────────────────────────────────────
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from api.config import API_TITLE, API_VERSION, API_DESCRIPTION
from api import db
from api.predictor.model_loader import load_models
from api.scheduler import start_scheduler, shutdown_scheduler
from api.routers import status, predictions, ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    logger.info("Ensuring MongoDB indexes...")
    db.init_indexes()
    logger.info("Loading models (this may take a moment)...")
    load_models()
    logger.info("Starting background scheduler...")
    start_scheduler()
    logger.info("Ready. Swagger UI → http://localhost:8000/docs")
    yield
    # SHUTDOWN
    logger.info("Shutting down scheduler...")
    shutdown_scheduler()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

# Allow all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(status.router)
app.include_router(predictions.router)
app.include_router(ohlcv.router)


# ── Root health check ─────────────────────────────────────────────────────────
@app.get("/", tags=["Health"], summary="Health check")
def health():
    return {"status": "ok", "service": API_TITLE, "version": API_VERSION}
