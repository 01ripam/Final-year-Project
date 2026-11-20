"""
pytorch.py
──────────
PyTorch Deep Learning Classifier implementation for the NIDS pipeline.

Implements a 3-layer Multilayer Perceptron (MLP) for network flow
classification. PyTorch models require input standardisation, so this
class also loads and applies a scikit-learn StandardScaler during inference.

Author : Person 2 — Deep Learning & XAI
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Optional

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from src.models.base import BaseClassifier, NUM_CLASSES
from src.models.dummy import DEFAULT_FEATURE_COLUMNS

logger = logging.getLogger("nids.models.pytorch")


_Base = nn.Module if nn else object
class MLPNetwork(_Base):
    """
    3-Layer MLP for tabular feature classification.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # logits
        return x


class PyTorchClassifier(BaseClassifier):
    """
    PyTorch model wrapper for the NIDS pipeline.
    
    Loads a saved state_dict (.pt) and a StandardScaler (.pkl).
    During inference, it standardises the input features, runs them through
    the MLP, and applies softmax to return class probabilities.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ) -> None:
        if torch is None:
            raise ImportError("torch is not installed. Run: pip install torch")

        self.name = "pytorch-mlp-v1.0"
        self.feature_columns = list(DEFAULT_FEATURE_COLUMNS)
        self.num_classes = NUM_CLASSES
        
        # Determine device (CPU for inference workers by default)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_: Optional[nn.Module] = None
        self.scaler: Optional[Any] = None

        if model_path and scaler_path:
            self.load(model_path, scaler_path)

    def load(self, model_path: str, scaler_path: Optional[str] = None) -> None:
        """
        Load PyTorch weights and the standard scaler.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
            
        if scaler_path and not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        try:
            # 1. Load scaler
            if scaler_path:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded scaler from %s", scaler_path)
            
            # 2. Init model architecture
            model = MLPNetwork(input_dim=len(self.feature_columns), num_classes=self.num_classes)
            
            # 3. Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.model_ = model
            logger.info("Loaded PyTorch model from %s on %s", model_path, self.device)
            
        except Exception as e:
            logger.exception("Failed to load PyTorch model from %s", model_path)
            raise e

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference using the PyTorch MLP.
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model not loaded. Call load(model_path, scaler_path) or pass paths to __init__."
            )

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Scale features if a scaler is loaded
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Convert to tensor
        x_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model_(x_tensor)
            proba = F.softmax(logits, dim=1)
            
        # Return as numpy array on CPU
        return proba.cpu().numpy().astype(np.float64)
