"""
verify_pipeline.py
──────────────────
Tests the ModelRegistry and the new models to ensure they can be loaded
and used for inference.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.models.registry import registry
from src.models.lightgbm import LightGBMClassifier
from src.models.pytorch import PyTorchClassifier
from src.models.dummy import DEFAULT_FEATURE_COLUMNS

def test_models():
    # Load LightGBM
    lgbm = LightGBMClassifier("models/lgbm-v1.pkl")
    registry.register(lgbm.name, lgbm)
    
    # Load PyTorch
    pytorch = PyTorchClassifier("models/pytorch-mlp-v1.pt", "models/scaler.pkl")
    registry.register(pytorch.name, pytorch)
    
    # Generate dummy feature vector
    features = np.random.rand(1, len(DEFAULT_FEATURE_COLUMNS))
    
    for name in registry.available:
        if name == "dummy-v1.0": continue
        
        registry.set_active(name)
        model = registry.active
        
        print(f"Testing {model.name}...")
        idx, conf, probs = model.predict(features)
        print(f"Predicted class index: {idx}")
        print(f"Confidence: {conf:.4f}")
        print(f"Probabilities shape: {len(probs)}")
        print("-" * 30)
        
    print("Verification complete!")

if __name__ == "__main__":
    test_models()
