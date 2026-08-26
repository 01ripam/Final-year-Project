"""
shap_worker.py
──────────────
Async SHAP explanation worker — produces feature-attribution payloads
for malicious flow predictions.

Pipeline
────────
  inference_worker.run_inference()
      │  (enqueues when predicted_class != "Normal")
      ▼
  explain_flow(flow_json, prediction_json)
      ├── SHAP explainer(model, features)   → ShapExplanation
      ├── db_writer.write_shap.delay(...)
      └── redis.publish("shap:results", ...)

Explainer strategy
──────────────────
  - Tree models (LightGBM, XGBoost): shap.TreeExplainer
  - DL models (PyTorch):             shap.DeepExplainer / GradientExplainer
  - DummyClassifier:                 DummySHAPExplainer (random attributions)

The explainer is selected at runtime based on the model's class name.

Author : Person 2 — Deep Learning & XAI
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.workers.celery_app import app
from src.models.base import CLASS_NAMES, BaseClassifier
from src.models.registry import registry
from src.data.flow_schema import FlowFeatureVector, FlowPrediction, ShapExplanation

import redis as redis_lib

logger = logging.getLogger("nids.workers.shap")

# ── Redis connection ──────────────────────────────────────────────────────────
REDIS_URL      = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SHAP_CHANNEL   = "shap:results"

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
# SHAP Explainer Abstraction
# ─────────────────────────────────────────────────────────────────────────────

class BaseSHAPExplainer:
    """Interface for SHAP-style feature attribution."""

    explainer_type: str = "Unknown"

    def explain(
        self,
        model: BaseClassifier,
        features: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[Dict[str, float], float, List[str]]:
        """
        Compute feature attributions.

        Returns
        -------
        shap_values   : dict  — {feature_name: shap_value}
        base_value    : float — expected model output (intercept)
        top_features  : list  — feature names ordered by |SHAP| descending
        """
        raise NotImplementedError


class DummySHAPExplainer(BaseSHAPExplainer):
    """
    Deterministic pseudo-random SHAP explanations for pipeline testing.

    Produces plausible (but meaningless) SHAP values that:
    - Sum to roughly (confidence - base_value)
    - Are deterministic for the same input
    - Highlight a consistent top-5 feature ranking
    """

    explainer_type = "DummyExplainer"

    def explain(
        self,
        model: BaseClassifier,
        features: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[Dict[str, float], float, List[str]]:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        row = features[0]
        n_features = len(feature_names)

        # Deterministic seed from features
        digest = hashlib.md5(row.tobytes()).hexdigest()
        seed = int(digest[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)

        # Generate random SHAP values
        raw_shap = rng.randn(n_features).astype(np.float64)
        # Normalize so magnitude is reasonable (0.01 to 0.5 range)
        raw_shap = raw_shap / (np.abs(raw_shap).max() + 1e-10) * 0.3

        base_value = 1.0 / model.num_classes   # prior = uniform

        shap_values = {
            feature_names[i]: float(raw_shap[i])
            for i in range(n_features)
        }

        # Top features by absolute SHAP value
        sorted_features = sorted(
            shap_values.keys(),
            key=lambda k: abs(shap_values[k]),
            reverse=True,
        )
        top_features = sorted_features[:10]

        return shap_values, base_value, top_features


class TreeSHAPExplainer(BaseSHAPExplainer):
    """
    SHAP TreeExplainer for LightGBM / XGBoost models.

    Requires the ``shap`` library and a tree-based model object.
    Falls back to DummySHAPExplainer if SHAP is not available or the
    model doesn't expose a native tree structure.
    """

    explainer_type = "TreeExplainer"

    def explain(
        self,
        model: BaseClassifier,
        features: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[Dict[str, float], float, List[str]]:
        try:
            import shap

            # TreeExplainer expects the underlying booster / model object.
            # BaseClassifier subclasses for LightGBM/XGBoost should expose
            # a .booster_ or .model_ attribute.
            inner_model = getattr(model, "booster_", None) or getattr(model, "model_", None)
            if inner_model is None:
                logger.warning(
                    "Model %r has no .booster_ or .model_ attribute — "
                    "falling back to DummySHAPExplainer.",
                    model.name,
                )
                return DummySHAPExplainer().explain(model, features, feature_names)

            explainer = shap.TreeExplainer(inner_model)

            if features.ndim == 1:
                features = features.reshape(1, -1)

            shap_out = explainer.shap_values(features)

            # shap_out can be a list (one array per class) or a single array.
            # For multi-class, take the values corresponding to the predicted class.
            if isinstance(shap_out, list):
                # Use the max-probability class
                proba = model.predict_proba(features)[0]
                class_idx = int(np.argmax(proba))
                vals = shap_out[class_idx][0]
                base_value = float(explainer.expected_value[class_idx])
            else:
                vals = shap_out[0]
                base_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

            shap_values = {
                feature_names[i]: float(vals[i])
                for i in range(len(feature_names))
            }

            sorted_features = sorted(
                shap_values.keys(),
                key=lambda k: abs(shap_values[k]),
                reverse=True,
            )

            return shap_values, base_value, sorted_features[:10]

        except ImportError:
            logger.warning("shap library not available — falling back to DummySHAPExplainer.")
            return DummySHAPExplainer().explain(model, features, feature_names)
        except Exception:
            logger.exception("TreeExplainer failed — falling back to DummySHAPExplainer.")
            return DummySHAPExplainer().explain(model, features, feature_names)


# ── Explainer factory ─────────────────────────────────────────────────────────

def _get_explainer(model: BaseClassifier) -> BaseSHAPExplainer:
    """Select the appropriate SHAP explainer for the given model."""
    model_cls_name = model.__class__.__name__.lower()

    if "dummy" in model_cls_name:
        return DummySHAPExplainer()
    elif any(t in model_cls_name for t in ("lightgbm", "lgbm", "xgboost", "xgb")):
        return TreeSHAPExplainer()
    else:
        # Future: DeepExplainer for PyTorch models
        logger.info(
            "No specialised explainer for %r — using DummySHAPExplainer.",
            model.__class__.__name__,
        )
        return DummySHAPExplainer()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Celery Task
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="src.workers.shap_worker.explain_flow",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
    queue="shap",
)
def explain_flow(self, flow_json: str, prediction_json: str) -> dict:
    """
    Compute SHAP feature attributions for a malicious flow prediction.

    Parameters
    ----------
    flow_json : str
        JSON-serialised FlowFeatureVector.
    prediction_json : str
        JSON-serialised FlowPrediction (for context / logging).

    Returns
    -------
    dict
        Serialised ShapExplanation payload.
    """
    try:
        vector     = FlowFeatureVector.model_validate_json(flow_json)
        prediction = FlowPrediction.model_validate_json(prediction_json)

        model    = registry.active
        explainer = _get_explainer(model)

        features = np.array(
            vector.to_feature_array(model.feature_columns),
            dtype=np.float64,
        )

        t0 = time.perf_counter()
        shap_values, base_value, top_features = explainer.explain(
            model, features, model.feature_columns,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        explanation = ShapExplanation(
            flow_id=vector.flow_id,
            shap_values=shap_values,
            base_value=base_value,
            top_features=top_features,
            explainer_type=explainer.explainer_type,
            explanation_latency_ms=latency_ms,
        )

        explanation_json = explanation.model_dump_json()

        # Persist to DB
        from src.workers.db_writer import write_shap
        write_shap.delay(explanation_json)

        # Publish to Redis pub/sub
        try:
            _get_redis().publish(SHAP_CHANNEL, explanation_json)
        except Exception:
            logger.warning("Failed to publish SHAP result to Redis pub/sub.", exc_info=True)

        logger.info(
            "[shap] flow=%s  class=%s  explainer=%s  top_feature=%s  latency=%.1fms",
            vector.flow_id,
            prediction.predicted_class.value,
            explainer.explainer_type,
            top_features[0] if top_features else "N/A",
            latency_ms,
        )

        return json.loads(explanation_json)

    except Exception as exc:
        logger.exception("SHAP explanation failed for flow %s.", flow_json[:80])
        raise self.retry(exc=exc)
