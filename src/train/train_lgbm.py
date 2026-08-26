"""
train_lgbm.py
─────────────
Trains a LightGBM model on the processed CICIDS2017 Parquet datasets.

Loads train.parquet, val.parquet, and test.parquet.
Evaluates using accuracy, F1 macro, and prints a confusion matrix.
Saves the trained model as a `.pkl` to `models/`.

Author : Person 1 — Data & Classical ML
Project: Enterprise NIDS — Week 2
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
try:
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
except ImportError:
    lgb = None

# Insert path to access project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.dummy import DEFAULT_FEATURE_COLUMNS
from src.data.label_mapper import UNIFIED_CLASSES, CLASS_TO_INT
from src.models.base import CLASS_NAMES

def load_data(filepath: str):
    df = pd.read_parquet(filepath)
    # Ensure correct column order
    X = df[DEFAULT_FEATURE_COLUMNS].values
    y = df["label"].values
    # Convert string labels to integer indices
    y = np.array([CLASS_TO_INT[label] for label in y])
    return X, y

def main():
    if lgb is None:
        print("Error: lightgbm or sklearn not installed. Run: pip install lightgbm scikit-learn")
        sys.exit(1)

    data_dir = Path("data/processed")
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        print("Error: Processed data not found. Run src/data/data_loader.py first.")
        sys.exit(1)

    print("Loading datasets...")
    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(val_path)
    X_test, y_test = load_data(test_path)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=64,
        objective="multiclass",
        num_class=len(CLASS_NAMES),
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 Macro: {f1:.4f}\n")
    
    print("Classification Report:")
    # Only include classes that are present in y_test
    present_classes = np.unique(np.concatenate((y_test, y_pred)))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=target_names))

    print("Saving model...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_out_path = models_dir / "lgbm-v1.pkl"
    with open(model_out_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {model_out_path}")

if __name__ == "__main__":
    main()
