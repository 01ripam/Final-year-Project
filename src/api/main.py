"""
main.py
───────
FastAPI application — ML inference REST API + real-time WebSocket alert server.

Endpoints
─────────
  GET  /health          — Readiness probe (Redis + DB connectivity)
  GET  /stats           — Pipeline statistics (queue lengths, active model)
  POST /predict         — On-demand single-flow inference
  GET  /models          — List registered models
  WS   /ws/alerts       — Real-time alert stream (subscribes to Redis pub/sub)

Start
─────
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Author : Person 4 — Platform & DevOps
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import redis as redis_lib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.base import CLASS_NAMES
from src.models.registry import registry
from src.models.dummy import DummyClassifier
from src.data.flow_schema import (
    FlowFeatureVector,
    FlowPrediction,
    AttackClass,
)

logger = logging.getLogger("nids.api")

# ── Configuration ─────────────────────────────────────────────────────────────
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FLOW_QUEUE_KEY = os.getenv("FLOW_QUEUE_KEY", "flow:queue")
ALERTS_CHANNEL = "alerts"

# ── Redis connections ─────────────────────────────────────────────────────────
_redis: Optional[redis_lib.Redis] = None


def _get_redis() -> redis_lib.Redis:
    global _redis
    if _redis is None:
        _redis = redis_lib.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis


# ─────────────────────────────────────────────────────────────────────────────
# Application Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown hooks."""
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("NIDS API starting up …")

    # Register default model if none present
    if len(registry) == 0:
        registry.register("dummy-v1.0", DummyClassifier())
        logger.info("Registered DummyClassifier as fallback model.")

    # Verify Redis connectivity
    try:
        _get_redis().ping()
        logger.info("Redis connected: %s", REDIS_URL)
    except Exception:
        logger.warning("Redis not reachable at %s — alerts will not broadcast.", REDIS_URL)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("NIDS API shutting down …")
    if _redis is not None:
        _redis.close()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise NIDS — ML Inference API",
    description=(
        "Real-time network intrusion detection system.\n\n"
        "Classifies network flows into 7 attack categories using ML models, "
        "with SHAP-based explainability and WebSocket alert streaming."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ── CORS (allow SOC dashboard frontend) ───────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus instrumentation (optional — graceful if not installed) ─────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
    logger.info("Prometheus instrumentation enabled at /metrics.")
except ImportError:
    logger.info("prometheus-fastapi-instrumentator not installed — /metrics disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Health & Stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health_check():
    """Readiness probe — checks Redis connectivity."""
    checks = {"redis": False, "model_loaded": False}

    try:
        _get_redis().ping()
        checks["redis"] = True
    except Exception:
        pass

    checks["model_loaded"] = len(registry) > 0

    healthy = all(checks.values())
    return {
        "status": "healthy" if healthy else "degraded",
        "checks": checks,
    }


@app.get("/stats", tags=["ops"])
def pipeline_stats():
    """Pipeline statistics — queue lengths, active model info."""
    stats = {
        "active_model": registry.active_name,
        "registered_models": registry.available,
        "flow_queue_length": 0,
    }

    try:
        stats["flow_queue_length"] = _get_redis().llen(FLOW_QUEUE_KEY)
    except Exception:
        pass

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Model Info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/models", tags=["models"])
def list_models():
    """List all registered models."""
    models = []
    for name in registry.available:
        model = registry.get(name)
        models.append({
            "name": name,
            "class": model.__class__.__name__,
            "num_features": len(model.feature_columns),
            "num_classes": model.num_classes,
            "is_active": name == registry.active_name,
        })
    return {"models": models}


# ─────────────────────────────────────────────────────────────────────────────
# On-Demand Prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict", tags=["inference"], response_model=dict)
def predict_flow(vector: FlowFeatureVector):
    """
    Run ML inference on a single FlowFeatureVector.

    Returns the FlowPrediction payload including class probabilities,
    confidence score, and model metadata.
    """
    try:
        model = registry.active
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = np.array(
        vector.to_feature_array(model.feature_columns),
        dtype=np.float64,
    ).reshape(1, -1)

    t0 = time.perf_counter()
    class_idx, confidence, class_probs = model.predict(features)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    predicted_class = CLASS_NAMES[class_idx]
    is_malicious = predicted_class != "Normal"

    prediction = FlowPrediction(
        flow_id=vector.flow_id,
        predicted_class=AttackClass(predicted_class),
        confidence=confidence,
        class_probabilities=class_probs,
        model_name=model.name,
        inference_latency_ms=latency_ms,
        shap_queued=is_malicious,
    )

    # Broadcast alert
    try:
        alert_payload = {
            "flow_id": vector.flow_id,
            "src_ip": vector.src_ip,
            "dst_ip": vector.dst_ip,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "is_alert": is_malicious,
        }
        _get_redis().publish(ALERTS_CHANNEL, json.dumps(alert_payload))
    except Exception:
        pass

    return json.loads(prediction.model_dump_json())


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — Real-time Alert Stream
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections for alert broadcasting."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Total: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(self.active_connections))

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


ws_manager = ConnectionManager()


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    Real-time alert stream.

    Subscribes to the Redis ``alerts`` pub/sub channel and forwards
    every message to the connected WebSocket client.
    """
    await ws_manager.connect(websocket)

    # Start Redis pub/sub listener in a background task
    pubsub = None
    try:
        r = _get_redis()
        pubsub = r.pubsub()
        pubsub.subscribe(ALERTS_CHANNEL)

        while True:
            # Non-blocking poll with short timeout
            message = pubsub.get_message(timeout=0.5)
            if message and message["type"] == "message":
                await websocket.send_text(message["data"])

            # Also check for client messages (ping/pong, close)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.1
                )
                # Echo back or handle client commands if needed
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error.")
    finally:
        ws_manager.disconnect(websocket)
        if pubsub is not None:
            try:
                pubsub.unsubscribe(ALERTS_CHANNEL)
                pubsub.close()
            except Exception:
                pass
