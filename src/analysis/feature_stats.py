# src/analysis/feature_stats.py
"""Feature statistics and importance analysis utilities for the NIDS synthetic dataset.

This module provides helper functions to:
- Compute per‑class descriptive statistics for each of the 78 canonical features.
- Generate a Pearson correlation matrix for the feature space.
- Train a lightweight LightGBM model to extract feature importances (gain based).
- Optionally export the results to CSV/JSON for downstream reporting.

The implementation is deliberately lightweight and has no side‑effects on the core
pipeline. It can be invoked as a script:

    python -m src.analysis.feature_stats \
        --input data/raw/synthetic_cicids.csv \
        --label-column "Label" \
        --output-dir reports/feature_stats_v1

Dependencies: pandas, numpy, lightgbm, scikit‑learn, tqdm (optional).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
import lightgbm as lgb

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("nids.feature_stats")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str, label_column: str = "Label") -> pd.DataFrame:
    """Load the synthetic CSV and ensure the label column exists."""
    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")
    return df


def per_class_statistics(df: pd.DataFrame, label_column: str = "Label") -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Compute mean, std and median for every numeric feature per class.

    Returns a nested dict of the form:
    {
        "Normal": {"flow_duration": (mean, std, median), ...},
        "DoS": {...},
        ...
    }
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    classes = df[label_column].unique()
    logger.info("Computing per‑class statistics for %d classes", len(classes))
    for cls in classes:
        cls_df = df[df[label_column] == cls]
        cls_stats: Dict[str, Tuple[float, float, float]] = {}
        for col in numeric_cols:
            values = cls_df[col].values
            mean = float(np.mean(values))
            std = float(np.std(values))
            median = float(np.median(values))
            cls_stats[col] = (mean, std, median)
        stats[cls] = cls_stats
    return stats


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix for all numeric features (excluding label)."""
    numeric_df = df.select_dtypes(include=[np.number])
    logger.info("Computing Pearson correlation matrix for %d numeric features", numeric_df.shape[1])
    return numeric_df.corr(method="pearson")


def lightgbm_feature_importance(
    df: pd.DataFrame,
    label_column: str = "Label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.Series:
    """Train a quick LightGBM classifier and return feature importances (gain)."""
    X = df.drop(columns=[label_column])
    y = df[label_column]
    y_enc, class_names = pd.factorize(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    logger.info(
        "Training LightGBM for feature importance (train %d, test %d)",
        X_train.shape[0],
        X_test.shape[0],
    )
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    params = {
        "objective": "multiclass",
        "num_class": len(class_names),
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": random_state,
    }
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    importance = pd.Series(model.feature_importance(importance_type="gain"), index=X.columns)
    return importance.sort_values(ascending=False)

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate feature statistics and importance for NIDS synthetic data")
    p.add_argument("--input", type=str, required=True, help="Path to synthetic CSV file")
    p.add_argument("--label-column", type=str, default="Label", help="Name of the label column")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to write reports (CSV/JSON)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    df = load_dataset(args.input, label_column=args.label_column)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per‑class stats
    stats = per_class_statistics(df, label_column=args.label_column)
    stats_file = out_dir / "per_class_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Per‑class statistics written to %s", stats_file)

    # Correlation matrix
    corr = correlation_matrix(df)
    corr_file = out_dir / "feature_correlation.csv"
    corr.to_csv(corr_file)
    logger.info("Correlation matrix written to %s", corr_file)

    # Feature importance
    importance = lightgbm_feature_importance(df, label_column=args.label_column)
    imp_file = out_dir / "feature_importance.csv"
    importance.to_csv(imp_file, header=["importance_gain"])
    logger.info("Feature importance written to %s", imp_file)

    logger.info("All analysis reports generated successfully.")

if __name__ == "__main__":
    main()
