"""
lightgbm.py
───────────
LightGBM Classifier implementation for the NIDS pipeline.

Wraps a standard ``lightgbm.Booster`` or ``LGBMClassifier``, mapping its
outputs to the unified ``BaseClassifier`` interface.  LightGBM is our
primary CPU-inference model due to its high throughput and native
support for TreeSHAP.

Author : Person 1 — Data & Classical ML
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import logging
import pickle
import os
from typing import Optional

import numpy as np
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.models.base import BaseClassifier, NUM_CLASSES
from src.models.dummy import DEFAULT_FEATURE_COLUMNS

logger = logging.getLogger("nids.models.lightgbm")


class LightGBMClassifier(BaseClassifier):
    """
    LightGBM model wrapper.

    The underlying model is expected to be an sklearn-API ``LGBMClassifier``
    saved via `pickle`.

    The model object is exposed as ``self.model_`` to allow
    ``shap.TreeExplainer`` to hook into it automatically in the SHAP worker.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        if lgb is None:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

        self.name = "lightgbm-v1.0"
        self.feature_columns = list(DEFAULT_FEATURE_COLUMNS)
        self.num_classes = NUM_CLASSES
        self.model_: Optional[lgb.LGBMClassifier] = None

        if model_path:
            self.load(model_path)

    def load(self, path: str) -> None:
        """
        Load a trained LGBMClassifier from a pickle file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            
            if not isinstance(model, lgb.LGBMClassifier):
                logger.warning(
                    "Loaded object is not an LGBMClassifier (got %s). "
                    "predict_proba may fail.", type(model)
                )

            self.model_ = model
            logger.info("Loaded LightGBM model from %s", path)
        except Exception as e:
            logger.exception("Failed to load LightGBM model from %s", path)
            raise e

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference using the loaded LightGBM model.
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model not loaded. Call load(path) or pass model_path to __init__."
            )

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # LightGBM predict_proba returns shape (N, num_classes)
        proba = self.model_.predict_proba(features)
        
        # Ensure it returns float64 array
        return np.asarray(proba, dtype=np.float64)

    @property
    def booster_(self):
        """
        Alias for shap.TreeExplainer compatibility.
        TreeExplainer looks for a .booster_ or .model_ attribute.
        """
        if hasattr(self.model_, "booster_"):
            return self.model_.booster_
        return self.model_
