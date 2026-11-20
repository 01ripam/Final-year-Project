"""
inference_worker.py
───────────────────
Celery worker that consumes FlowFeatureVector payloads from Redis ``flow:queue``,
runs ML inference via the model registry, and dispatches downstream tasks
(SHAP explanation, DB persistence, WebSocket alerts).

Pipeline
────────
  Redis flow:queue (BRPOP)
      │
      ▼
  run_inference(flow_json)
      ├── model.predict(features)       → FlowPrediction
      ├── if malicious → shap_worker.explain_flow.delay(...)
      ├── db_writer.write_flow.delay(...)
      ├── db_writer.write_prediction.delay(...)
      └── redis.publish("alerts", alert_json)   ← for WebSocket broadcast

  consume_flow_queue()
      └── Continuous BRPOP loop → dispatches run_inference for each flow

Author : Person 1 — Data & Classical ML  /  Person 2 — Deep Learning
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import redis as redis_lib

# ── Path setup for cross-module imports ───────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.workers.celery_app import app
from src.models.base import CLASS_NAMES
from src.models.registry import registry, load_default_models
from src.models.dummy import DummyClassifier
from src.data.flow_schema import FlowFeatureVector, FlowPrediction, AttackClass

logger = logging.getLogger("nids.workers.inference")

# ── Redis connection for pub/sub alerts ───────────────────────────────────────
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
FLOW_QUEUE_KEY = os.getenv("FLOW_QUEUE_KEY", "flow:queue")
ALERTS_CHANNEL = "alerts"

_redis: Optional[redis_lib.Redis] = None


def _get_redis() -> redis_lib.Redis:
    """Lazy singleton Redis connection."""
    global _redis
    if _redis is None:
        _redis = redis_lib.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis


def _ensure_model_registered() -> None:
    """Discover trained models from disk or fall back to DummyClassifier."""
    if len(registry) == 0:
        load_default_models()
        logger.info(
            "Registered models in inference worker: %s (active: %s)",
            registry.available,
            registry.active_name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core Inference Task
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="src.workers.inference_worker.run_inference",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
)
def run_inference(self, flow_json: str) -> dict:
    """
    Run ML inference on a single FlowFeatureVector JSON payload.

    Parameters
    ----------
    flow_json : str
        JSON-serialised FlowFeatureVector (from Redis flow:queue or REST API).

    Returns
    -------
    dict
        Serialised FlowPrediction payload.
    """
    _ensure_model_registered()

    try:
        # 1. Deserialise the flow vector
        vector = FlowFeatureVector.model_validate_json(flow_json)
        logger.debug("Inference start — flow_id=%s", vector.flow_id)

        # 2. Extract feature array aligned to model's expected column order
        model = registry.active
        features = np.array(
            vector.to_feature_array(model.feature_columns),
            dtype=np.float64,
        ).reshape(1, -1)

        # 3. Run inference
        t0 = time.perf_counter()
        class_idx, confidence, class_probs = model.predict(features)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        predicted_class = CLASS_NAMES[class_idx]

        # 4. Build FlowPrediction
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

        prediction_json = prediction.model_dump_json()

        # 5. Dispatch downstream tasks (import here to avoid circular imports)
        from src.workers.db_writer import write_flow, write_prediction

        write_flow.delay(flow_json)
        write_prediction.delay(prediction_json)

        # 6. If malicious → enqueue SHAP explanation
        if is_malicious:
            from src.workers.shap_worker import explain_flow
            explain_flow.delay(flow_json, prediction_json)

        # 7. Publish alert to Redis pub/sub for WebSocket broadcast
        try:
            alert_payload = {
                "flow_id":         vector.flow_id,
                "src_ip":          vector.src_ip,
                "dst_ip":          vector.dst_ip,
                "src_port":        vector.src_port,
                "dst_port":        vector.dst_port,
                "protocol":        vector.protocol,
                "predicted_class": predicted_class,
                "confidence":      round(confidence, 4),
                "model_name":      model.name,
                "latency_ms":      round(latency_ms, 2),
                "is_alert":        is_malicious,
            }
            _get_redis().publish(ALERTS_CHANNEL, json.dumps(alert_payload))
        except Exception:
            logger.warning("Failed to publish alert to Redis pub/sub.", exc_info=True)

        logger.info(
            "[inference] flow=%s  class=%s  conf=%.4f  latency=%.1fms  alert=%s",
            vector.flow_id, predicted_class, confidence, latency_ms, is_malicious,
        )

        return json.loads(prediction_json)

    except Exception as exc:
        logger.exception("Inference failed for flow payload.")
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────────────────────────────────────
# Continuous Queue Consumer
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="src.workers.inference_worker.consume_flow_queue",
    bind=True,
    acks_late=True,
)
def consume_flow_queue(self, max_flows: Optional[int] = None) -> dict:
    """
    Continuously BRPOP from ``flow:queue`` and dispatch ``run_inference``
    for each flow vector.

    This is a long-running task meant to be started once.  It blocks on
    BRPOP and processes flows sequentially within this task, but since
    ``run_inference`` is dispatched as a separate Celery task, actual
    inference can be parallelised across the worker pool.

    Parameters
    ----------
    max_flows : int, optional
        Stop after consuming this many flows (for testing).
        None = run indefinitely.

    Returns
    -------
    dict
        ``{"consumed": N}`` — number of flows dispatched.
    """
    _ensure_model_registered()
    r = _get_redis()
    consumed = 0

    logger.info(
        "Consumer started — queue=%s  max_flows=%s",
        FLOW_QUEUE_KEY, max_flows or "∞",
    )

    while True:
        if max_flows is not None and consumed >= max_flows:
            break

        # BRPOP blocks until a flow vector is available (5s timeout for
        # periodic loop-continue so we can check stop conditions).
        result = r.brpop(FLOW_QUEUE_KEY, timeout=5)
        if result is None:
            continue   # timeout — loop back

        _key, flow_json = result
        run_inference.delay(flow_json)
        consumed += 1

        if consumed % 100 == 0:
            logger.info("Consumer dispatched %d flows.", consumed)

    logger.info("Consumer stopped after %d flows.", consumed)
    return {"consumed": consumed}
