"""
registry.py
───────────
Model registry — singleton that loads, stores, and serves ML classifiers.

The inference worker calls ``registry.get(model_name)`` to obtain the
active classifier.  Models are registered at startup; switching the
active model is a single ``registry.set_active(name)`` call with no
pipeline restart required.

Usage
─────
    from src.models.registry import registry

    # At startup
    registry.register("dummy", DummyClassifier())
    registry.set_active("dummy")

    # In inference worker
    model = registry.active
    class_idx, confidence, probs = model.predict(features)

    # Later, when a real model is trained:
    registry.register("lightgbm-v1", LightGBMClassifier("models/lgbm.txt"))
    registry.set_active("lightgbm-v1")

Author : Person 1 — Data & Classical ML
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from src.models.base import BaseClassifier

logger = logging.getLogger("nids.models.registry")


class ModelRegistry:
    """
    Thread-safe model registry.

    Guarantees
    ──────────
    - ``register()`` / ``get()`` / ``set_active()`` are all thread-safe.
    - ``active`` returns the currently selected model (or raises if none set).
    - Multiple models can be registered simultaneously (useful for A/B testing
      or ensemble voting).
    """

    def __init__(self) -> None:
        self._models: Dict[str, BaseClassifier] = {}
        self._active_name: Optional[str] = None
        self._lock = threading.RLock()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, model: BaseClassifier) -> None:
        """
        Register a model under the given name.

        If a model with the same name already exists, it is replaced
        (useful for hot-reloading updated weights).
        """
        with self._lock:
            model.name = name
            self._models[name] = model
            logger.info("Registered model %r: %r", name, model)

            # Auto-activate the first model registered
            if self._active_name is None:
                self._active_name = name
                logger.info("Auto-activated model %r (first registered).", name)

    def unregister(self, name: str) -> None:
        """Remove a model from the registry."""
        with self._lock:
            self._models.pop(name, None)
            if self._active_name == name:
                self._active_name = None
                logger.warning(
                    "Unregistered active model %r — no active model set.", name
                )

    # ── Selection ─────────────────────────────────────────────────────────────

    def set_active(self, name: str) -> None:
        """Set the active model by name.  Raises KeyError if not registered."""
        with self._lock:
            if name not in self._models:
                raise KeyError(
                    f"Model {name!r} not registered. "
                    f"Available: {list(self._models.keys())}"
                )
            self._active_name = name
            logger.info("Active model set to %r.", name)

    @property
    def active(self) -> BaseClassifier:
        """
        Return the currently active model.

        Raises
        ------
        RuntimeError
            If no model has been registered or activated.
        """
        with self._lock:
            if self._active_name is None or self._active_name not in self._models:
                raise RuntimeError(
                    "No active model.  Call registry.register() and "
                    "registry.set_active() first."
                )
            return self._models[self._active_name]

    @property
    def active_name(self) -> Optional[str]:
        """Name of the currently active model (None if not set)."""
        with self._lock:
            return self._active_name

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> BaseClassifier:
        """Retrieve a model by name.  Raises KeyError if not found."""
        with self._lock:
            if name not in self._models:
                raise KeyError(
                    f"Model {name!r} not found. "
                    f"Available: {list(self._models.keys())}"
                )
            return self._models[name]

    @property
    def available(self) -> List[str]:
        """List of all registered model names."""
        with self._lock:
            return list(self._models.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"<ModelRegistry models={list(self._models.keys())} "
                f"active={self._active_name!r}>"
            )


# ── Module-level singleton ────────────────────────────────────────────────────
# Import ``registry`` from this module — do NOT instantiate your own.

registry = ModelRegistry()
