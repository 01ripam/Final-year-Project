"""
train_pytorch.py
────────────────
Trains a PyTorch MLP model on the processed CICIDS2017 Parquet datasets.

Standardises the input features, trains a 3-layer MLP using CrossEntropyLoss
and Adam, evaluates on the test set, and saves both the model weights (.pt)
and the scaler (.pkl) to `models/`.

Author : Person 2 — Deep Learning & XAI
Project: Enterprise NIDS — Week 2
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, classification_report
except ImportError:
    torch = None

# Insert path to access project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.dummy import DEFAULT_FEATURE_COLUMNS
from src.models.base import CLASS_NAMES, NUM_CLASSES
from src.models.pytorch import MLPNetwork
from src.data.label_mapper import CLASS_TO_INT

def load_data(filepath: str):
    df = pd.read_parquet(filepath)
    X = df[DEFAULT_FEATURE_COLUMNS].values
    y = df["label"].values
    y = np.array([CLASS_TO_INT[label] for label in y])
    return X, y

def main():
    if torch is None:
        print("Error: torch or sklearn not installed. Run: pip install torch scikit-learn")
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

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    scaler_path = models_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    batch_size = 256
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    print("Initialising MLP...")
    model = MLPNetwork(input_dim=len(DEFAULT_FEATURE_COLUMNS), num_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0

    print("Training PyTorch Classifier...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Evaluating on test set...")
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1 Macro: {f1:.4f}\n")
    
    print("Classification Report:")
    present_classes = np.unique(np.concatenate((y_test, preds)))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    print(classification_report(y_test, preds, labels=present_classes, target_names=target_names))

    model_out_path = models_dir / "pytorch-mlp-v1.pt"
    torch.save(best_model_state, model_out_path)
    print(f"Model saved to {model_out_path}")

if __name__ == "__main__":
    main()
