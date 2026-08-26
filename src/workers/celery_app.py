"""
celery_app.py
─────────────
Shared Celery application instance for all NIDS workers.

Queue topology
──────────────
  inference   — high-priority, latency-sensitive flow classification
  shap        — lower-priority, compute-heavy SHAP explanations
  db_write    — batched, throughput-oriented TimescaleDB writes

Start workers
─────────────
  # All queues in one process (dev / single-node):
  celery -A src.workers.celery_app worker -Q inference,shap,db_write -l INFO

  # Production (separate worker pools):
  celery -A src.workers.celery_app worker -Q inference  -c 4  -l INFO
  celery -A src.workers.celery_app worker -Q shap       -c 2  -l INFO
  celery -A src.workers.celery_app worker -Q db_write   -c 2  -l INFO

Author : Person 4 — Platform & DevOps
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import os

from celery import Celery

# ── Broker / backend URLs from environment (fall back to Docker defaults) ─────
REDIS_BROKER_URL  = os.getenv("CELERY_BROKER_URL",  "redis://localhost:6379/0")
REDIS_BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# ── Create the Celery application ─────────────────────────────────────────────

app = Celery(
    "nids",
    broker=REDIS_BROKER_URL,
    backend=REDIS_BACKEND_URL,
)

# ── Configuration ─────────────────────────────────────────────────────────────

app.conf.update(
    # Serialisation — JSON everywhere for cross-language debuggability
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task acknowledgement — acks_late + reject_on_worker_lost gives
    # at-least-once delivery guarantees.
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Prefetch — keep it low so work is distributed evenly
    worker_prefetch_multiplier=2,

    # Concurrency defaults (overridable with -c flag)
    worker_concurrency=4,

    # Result expiry — keep results for 1 hour (mostly for debugging)
    result_expires=3600,

    # Task-level settings
    task_track_started=True,
    task_time_limit=300,            # hard kill after 5 min
    task_soft_time_limit=240,       # soft warning at 4 min

    # Queue routing
    task_routes={
        "src.workers.inference_worker.*": {"queue": "inference"},
        "src.workers.shap_worker.*":     {"queue": "shap"},
        "src.workers.db_writer.*":       {"queue": "db_write"},
    },

    # Default queue for unrouted tasks
    task_default_queue="inference",
)

# ── Auto-discover tasks from the workers package ──────────────────────────────
app.autodiscover_tasks(["src.workers"])
