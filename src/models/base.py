"""
base.py
───────
Abstract base class for all ML classifiers in the NIDS pipeline.

Every model (LightGBM, XGBoost, PyTorch, etc.) implements this interface
so that the inference worker can swap models without touching any pipeline code.

Author : Person 1 — Data & Classical ML  /  Person 2 — Deep Learning
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


# 7-class taxonomy — mirrors src/data/label_mapper.py and flow_schema.AttackClass
CLASS_NAMES: List[str] = [
    "Normal",
    "DoS",
    "DDoS",
    "Port Scan",
    "Brute Force",
    "Botnet",
    "Web Attack",
]

NUM_CLASSES: int = len(CLASS_NAMES)


class BaseClassifier(ABC):
    """
    Contract that every NIDS classifier must implement.

    Subclass responsibilities
    ─────────────────────────
    1. Set ``self.name`` to a unique model identifier (e.g. ``"lightgbm-v1.2"``).
    2. Set ``self.feature_columns`` to the ordered list of feature names the
       model was trained on.  The inference worker calls
       ``FlowFeatureVector.to_feature_array(model.feature_columns)`` to
       guarantee column alignment.
    3. Implement ``predict_proba()`` — the only required method.

    The base class provides a convenience ``predict()`` that derives the
    argmax class index, confidence score, and per-class probability dict
    from ``predict_proba()``.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    name: str = "unnamed-model"
    """Unique model identifier used in FlowPrediction.model_name."""

    feature_columns: List[str] = []
    """
    Ordered feature names the model expects as input.
    Must match the training-time column order exactly.
    """

    num_classes: int = NUM_CLASSES
    """Number of output classes (default 7)."""

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates for one or more samples.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, F)`` where ``N`` is the batch size and ``F`` is
            ``len(self.feature_columns)``.

        Returns
        -------
        np.ndarray
            Shape ``(N, num_classes)`` with values in [0, 1] that sum to ~1
            per row.
        """
        ...

    # ── Convenience methods ───────────────────────────────────────────────────

    def predict(
        self,
        features: np.ndarray,
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Single-sample convenience wrapper around ``predict_proba()``.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(1, F)`` or ``(F,)`` — a single flow's feature vector.

        Returns
        -------
        class_idx   : int          — predicted class index (0–6)
        confidence  : float        — probability of the predicted class
        class_probs : dict         — {class_name: probability} for all classes
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.predict_proba(features)     # (1, 7)
        row   = proba[0]

        class_idx  = int(np.argmax(row))
        confidence = float(row[class_idx])

        class_probs = {
            CLASS_NAMES[i]: float(row[i]) for i in range(self.num_classes)
        }

        return class_idx, confidence, class_probs

    def predict_batch(
        self,
        features: np.ndarray,
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """
        Multi-sample convenience wrapper.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, F)``.

        Returns
        -------
        list of (class_idx, confidence, class_probs) tuples — one per sample.
        """
        proba = self.predict_proba(features)
        results = []
        for row in proba:
            idx  = int(np.argmax(row))
            conf = float(row[idx])
            probs = {CLASS_NAMES[i]: float(row[i]) for i in range(self.num_classes)}
            results.append((idx, conf, probs))
        return results

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    def load(self, path: str) -> None:
        """
        Optional — load model weights / artefacts from disk.
        Override in subclasses that need it.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"features={len(self.feature_columns)} classes={self.num_classes}>"
        )
