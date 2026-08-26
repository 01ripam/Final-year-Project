"""
db_writer.py
────────────
Async TimescaleDB bulk-write worker.

Provides Celery tasks that persist flow vectors, predictions, and SHAP
explanations to the three TimescaleDB hypertables defined in init.sql.

Connection management
─────────────────────
  Uses synchronous ``psycopg2`` inside Celery tasks (simpler than managing
  an asyncpg event loop in Celery's prefork worker).  For FastAPI's own
  async endpoints, use asyncpg directly.

  A module-level connection pool is lazily initialised on first use.

Tables
──────
  flows              — raw flow feature vectors (features as JSONB)
  predictions        — ML classification results
  shap_explanations  — SHAP feature attributions

Author : Person 4 — Platform & DevOps
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.workers.celery_app import app
from src.data.flow_schema import FlowFeatureVector, FlowPrediction, ShapExplanation

logger = logging.getLogger("nids.workers.db_writer")

# ── Database connection settings ──────────────────────────────────────────────
DB_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
DB_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME     = os.getenv("POSTGRES_DB",       "nids_db")
DB_USER     = os.getenv("POSTGRES_USER",     "nids")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "nids_secret")

_pool = None


def _get_connection():
    """
    Get a connection from the lazy-initialised connection pool.

    Uses psycopg2 (sync) for Celery worker compatibility.
    Falls back gracefully if the DB is unreachable (logs warning, returns None).
    """
    global _pool

    if _pool is None:
        try:
            import psycopg2
            from psycopg2 import pool as pg_pool

            _pool = pg_pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            logger.info("DB connection pool created: %s@%s:%d/%s", DB_USER, DB_HOST, DB_PORT, DB_NAME)
        except ImportError:
            logger.warning(
                "psycopg2 not installed — DB writes will be logged but not persisted. "
                "Install with: pip install psycopg2-binary"
            )
            return None
        except Exception:
            logger.warning("Could not connect to TimescaleDB — writes will be skipped.", exc_info=True)
            return None

    try:
        return _pool.getconn()
    except Exception:
        logger.warning("Could not acquire DB connection from pool.", exc_info=True)
        return None


def _release_connection(conn) -> None:
    """Return a connection to the pool."""
    if _pool is not None and conn is not None:
        try:
            _pool.putconn(conn)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns to exclude from the JSONB features blob
# ─────────────────────────────────────────────────────────────────────────────

_FLOW_IDENTITY_COLS = {
    "flow_id", "capture_timestamp",
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "sensor_id", "network_segment",
}


# ─────────────────────────────────────────────────────────────────────────────
# Write Flow Task
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="src.workers.db_writer.write_flow",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
    queue="db_write",
)
def write_flow(self, flow_json: str) -> dict:
    """
    Persist a FlowFeatureVector to the ``flows`` hypertable.

    The full feature vector is stored as JSONB in the ``features`` column
    while identity/metadata columns are stored in dedicated typed columns.
    """
    try:
        vector = FlowFeatureVector.model_validate_json(flow_json)
        record = vector.to_db_record()

        conn = _get_connection()
        if conn is None:
            logger.debug("[db] flow=%s — DB unavailable, skipping write.", vector.flow_id)
            return {"status": "skipped", "flow_id": vector.flow_id}

        try:
            # Build JSONB features blob (everything except identity columns)
            features_blob = {
                k: v for k, v in record.items()
                if k not in _FLOW_IDENTITY_COLS
            }

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO flows (
                        flow_id, capture_timestamp,
                        src_ip, dst_ip, src_port, dst_port, protocol,
                        flow_duration, total_fwd_packets, total_bwd_packets,
                        sensor_id, network_segment, features
                    ) VALUES (
                        %(flow_id)s, %(capture_timestamp)s,
                        %(src_ip)s, %(dst_ip)s, %(src_port)s, %(dst_port)s, %(protocol)s,
                        %(flow_duration)s, %(total_fwd_packets)s, %(total_bwd_packets)s,
                        %(sensor_id)s, %(network_segment)s, %(features)s
                    )
                    ON CONFLICT (flow_id, capture_timestamp) DO NOTHING
                    """,
                    {
                        "flow_id":           record["flow_id"],
                        "capture_timestamp": record["capture_timestamp"],
                        "src_ip":            record["src_ip"],
                        "dst_ip":            record["dst_ip"],
                        "src_port":          record["src_port"],
                        "dst_port":          record["dst_port"],
                        "protocol":          record["protocol"],
                        "flow_duration":     record.get("flow_duration"),
                        "total_fwd_packets": record.get("total_fwd_packets"),
                        "total_bwd_packets": record.get("total_bwd_packets"),
                        "sensor_id":         record.get("sensor_id"),
                        "network_segment":   record.get("network_segment"),
                        "features":          json.dumps(features_blob),
                    },
                )
            conn.commit()
            logger.debug("[db] flow=%s written to flows table.", vector.flow_id)
            return {"status": "ok", "flow_id": vector.flow_id}

        finally:
            _release_connection(conn)

    except Exception as exc:
        logger.exception("Failed to write flow to DB.")
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────────────────────────────────────
# Write Prediction Task
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="src.workers.db_writer.write_prediction",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
    queue="db_write",
)
def write_prediction(self, prediction_json: str) -> dict:
    """Persist a FlowPrediction to the ``predictions`` hypertable."""
    try:
        prediction = FlowPrediction.model_validate_json(prediction_json)

        conn = _get_connection()
        if conn is None:
            logger.debug("[db] prediction flow=%s — DB unavailable, skipping.", prediction.flow_id)
            return {"status": "skipped", "flow_id": prediction.flow_id}

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions (
                        flow_id, inference_timestamp,
                        predicted_class, confidence, class_probabilities,
                        model_name, inference_latency_ms, shap_queued
                    ) VALUES (
                        %(flow_id)s, %(inference_timestamp)s,
                        %(predicted_class)s, %(confidence)s, %(class_probabilities)s,
                        %(model_name)s, %(inference_latency_ms)s, %(shap_queued)s
                    )
                    ON CONFLICT (flow_id, inference_timestamp) DO NOTHING
                    """,
                    {
                        "flow_id":              prediction.flow_id,
                        "inference_timestamp":  prediction.inference_timestamp.isoformat(),
                        "predicted_class":      prediction.predicted_class.value if hasattr(prediction.predicted_class, 'value') else str(prediction.predicted_class),
                        "confidence":           prediction.confidence,
                        "class_probabilities":  json.dumps(prediction.class_probabilities),
                        "model_name":           prediction.model_name,
                        "inference_latency_ms": prediction.inference_latency_ms,
                        "shap_queued":          prediction.shap_queued,
                    },
                )
            conn.commit()
            logger.debug("[db] prediction flow=%s written.", prediction.flow_id)
            return {"status": "ok", "flow_id": prediction.flow_id}

        finally:
            _release_connection(conn)

    except Exception as exc:
        logger.exception("Failed to write prediction to DB.")
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────────────────────────────────────
# Write SHAP Explanation Task
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="src.workers.db_writer.write_shap",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
    queue="db_write",
)
def write_shap(self, shap_json: str) -> dict:
    """Persist a ShapExplanation to the ``shap_explanations`` hypertable."""
    try:
        explanation = ShapExplanation.model_validate_json(shap_json)

        conn = _get_connection()
        if conn is None:
            logger.debug("[db] shap flow=%s — DB unavailable, skipping.", explanation.flow_id)
            return {"status": "skipped", "flow_id": explanation.flow_id}

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO shap_explanations (
                        flow_id, explanation_timestamp,
                        shap_values, base_value, top_features,
                        explainer_type, explanation_latency_ms
                    ) VALUES (
                        %(flow_id)s, %(explanation_timestamp)s,
                        %(shap_values)s, %(base_value)s, %(top_features)s,
                        %(explainer_type)s, %(explanation_latency_ms)s
                    )
                    ON CONFLICT (flow_id, explanation_timestamp) DO NOTHING
                    """,
                    {
                        "flow_id":              explanation.flow_id,
                        "explanation_timestamp": explanation.explanation_timestamp.isoformat(),
                        "shap_values":          json.dumps(explanation.shap_values),
                        "base_value":           explanation.base_value,
                        "top_features":         explanation.top_features,
                        "explainer_type":       explanation.explainer_type,
                        "explanation_latency_ms": explanation.explanation_latency_ms,
                    },
                )
            conn.commit()
            logger.debug("[db] shap flow=%s written.", explanation.flow_id)
            return {"status": "ok", "flow_id": explanation.flow_id}

        finally:
            _release_connection(conn)

    except Exception as exc:
        logger.exception("Failed to write SHAP explanation to DB.")
        raise self.retry(exc=exc)


# ─────────────────────────────────────────────────────────────────────────────
# Batch Write (for bulk imports / replays)
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="src.workers.db_writer.write_batch",
    bind=True,
    max_retries=2,
    default_retry_delay=15,
    acks_late=True,
    queue="db_write",
)
def write_batch(self, records_json: List[str], table: str) -> dict:
    """
    Batch-insert multiple records into the specified table.

    Parameters
    ----------
    records_json : list[str]
        List of JSON-serialised payloads (FlowFeatureVector, FlowPrediction,
        or ShapExplanation depending on ``table``).
    table : str
        One of: ``"flows"``, ``"predictions"``, ``"shap_explanations"``.

    Returns
    -------
    dict
        ``{"status": "ok", "inserted": N}``
    """
    task_map = {
        "flows":             write_flow,
        "predictions":       write_prediction,
        "shap_explanations": write_shap,
    }

    if table not in task_map:
        raise ValueError(f"Unknown table {table!r}. Must be one of {list(task_map.keys())}.")

    writer = task_map[table]
    inserted = 0

    for record_json in records_json:
        try:
            writer(record_json)   # direct call, not .delay() — same process
            inserted += 1
        except Exception:
            logger.exception("Batch write failed for record in %s.", table)

    logger.info("[db] batch write to %s: %d / %d succeeded.", table, inserted, len(records_json))
    return {"status": "ok", "inserted": inserted, "total": len(records_json)}
