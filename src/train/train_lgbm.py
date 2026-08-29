"""
train_lgbm.py
─────────────
Optimised LightGBM Training & Imbalance Handling Pipeline for Enterprise NIDS.

Features:
  1. Data Quality & Distribution Diagnostic Report.
  2. Strict 78-feature canonical contract enforcement.
  3. Multi-experiment Imbalance Benchmark:
       - Baseline LightGBM (unweighted)
       - Balanced Class Weights LightGBM
       - Tuned Regularised LightGBM + Class Weights
       - Controlled Random Oversampling (Train-only) + LightGBM
       - SMOTE (Train-only) + LightGBM
  4. Model Selection strictly on Validation Macro F1.
  5. Test Set Evaluation & Per-Class Performance Breakdown.
  6. Feature Importance extraction (Gain and Split count).
  7. Export of artefacts: lgbm-v2.pkl, class_weights.json, feature_names.json,
     label_encoder.pkl, and experiment_results.csv.

Author : Person 1 — Data & Classical ML
Project: Enterprise NIDS — Production Optimisation
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

try:
    # pyrefly: ignore [missing-import]
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import RandomOverSampler, SMOTE
except ImportError:
    RandomOverSampler = None
    SMOTE = None

# Insert path for project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.label_mapper import CLASS_TO_INT, UNIFIED_CLASSES
from src.models.base import CLASS_NAMES, NUM_CLASSES


def load_feature_names(feature_file: Path) -> List[str]:
    """Load canonical feature list."""
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature column definition missing: {feature_file}")
    with open(feature_file, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    if len(feature_cols) != 78:
        raise ValueError(f"Expected 78 canonical features, got {len(feature_cols)}")
    return feature_cols


def run_diagnostic_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> None:
    """Print thorough diagnostic report of datasets."""
    print("=" * 80)
    print("DATASET DIAGNOSTIC & DATA QUALITY REPORT")
    print("=" * 80)
    
    splits = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]
    
    for name, df in splits:
        total = len(df)
        print(f"\n--- {name} Set: {total:,} rows × {len(feature_cols)} features ---")
        counts = df["label"].value_counts()
        for cls_name in CLASS_NAMES:
            cnt = counts.get(cls_name, 0)
            pct = (cnt / total) * 100.0 if total > 0 else 0.0
            print(f"  {cls_name:<15}: {cnt:>6,}  ({pct:6.2f}%)")
            
        # Check nulls and infs
        num_nulls = df[feature_cols].isna().sum().sum()
        num_infs = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
        print(f"  Data Quality Checks: Missing={num_nulls:,} | Inf={num_infs:,}")

    # Overlap / Leakage Check
    train_set = set(map(tuple, train_df[feature_cols].values))
    val_set = set(map(tuple, val_df[feature_cols].values))
    test_set = set(map(tuple, test_df[feature_cols].values))
    
    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))
    
    print("\n--- Cross-Split Duplication & Isolation ---")
    print(f"  Train/Val Overlap:  {train_val_overlap} rows")
    print(f"  Train/Test Overlap: {train_test_overlap} rows")
    print(f"  Val/Test Overlap:   {val_test_overlap} rows")
    
    if train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0:
        print("\nDATA LEAKAGE CHECK: PASS (Strict split isolation confirmed)")
    else:
        print("\nDATA LEAKAGE CHECK: WARNING (Overlapping samples detected)")
    print("=" * 80 + "\n")


def calculate_balanced_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Calculate balanced class weights from training labels only."""
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_and_eval(
    name: str,
    model: Any,






















    
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray = None
) -> Dict[str, Any]:
    """Fit model and evaluate on validation and test sets."""
    fit_params = {}
    if sample_weight is not None:
        fit_params["sample_weight"] = sample_weight
        
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        **fit_params
    )
    
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    val_weighted_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
    val_bal_acc = balanced_accuracy_score(y_val, val_preds)
    
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    test_bal_acc = balanced_accuracy_score(y_test, test_preds)
    
    best_iter = getattr(model, "best_iteration_", model.n_estimators)
    
    return {
        "Experiment": name,
        "Val Accuracy": val_acc,
        "Val Macro F1": val_macro_f1,
        "Val Weighted F1": val_weighted_f1,
        "Val Balanced Acc": val_bal_acc,
        "Test Accuracy": test_acc,
        "Test Macro F1": test_macro_f1,
        "Test Weighted F1": test_weighted_f1,
        "Test Balanced Acc": test_bal_acc,
        "Best Iteration": best_iter,
        "_model": model,
        "_val_preds": val_preds,
        "_test_preds": test_preds
    }


def main():
    if lgb is None:
        print("Error: lightgbm is not installed. Run: pip install lightgbm")
        sys.exit(1)

    data_dir = Path("data/processed")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    feature_file = data_dir / "feature_columns.txt"
    feature_cols = load_feature_names(feature_file)

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        print("Error: Processed parquet datasets missing from data/processed/")
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)

    # 1. Run Diagnostic Report
    run_diagnostic_report(train_df, val_df, test_df, feature_cols)

    # 2. Extract strictly ordered feature matrices and target vectors
    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = np.array([CLASS_TO_INT[label] for label in train_df["label"]], dtype=np.int64)

    X_val = val_df[feature_cols].values.astype(np.float64)
    y_val = np.array([CLASS_TO_INT[label] for label in val_df["label"]], dtype=np.int64)

    X_test = test_df[feature_cols].values.astype(np.float64)
    y_test = np.array([CLASS_TO_INT[label] for label in test_df["label"]], dtype=np.int64)

    # 3. Calculate Class Weights (Training set only)
    cw_dict = calculate_balanced_class_weights(y_train)
    print("Calculated Training Set Balanced Class Weights:")
    for c_idx, w in cw_dict.items():
        print(f"  {CLASS_NAMES[c_idx]:<15} (Class {c_idx}): weight = {w:.4f}")
    print()

    experiments = []

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment 1: Baseline LightGBM (unweighted)
    # ─────────────────────────────────────────────────────────────────────────
    print("Running Experiment 1: Baseline LightGBM...")
    model_exp1 = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=64,
        objective="multiclass",
        num_class=NUM_CLASSES,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True
    )
    res1 = train_and_eval("Baseline", model_exp1, X_train, y_train, X_val, y_val, X_test, y_test)
    experiments.append(res1)

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment 2: LightGBM + Balanced Class Weights
    # ─────────────────────────────────────────────────────────────────────────
    print("Running Experiment 2: LightGBM + Balanced Class Weights...")
    model_exp2 = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=64,
        objective="multiclass",
        num_class=NUM_CLASSES,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True
    )
    res2 = train_and_eval("Class Weights", model_exp2, X_train, y_train, X_val, y_val, X_test, y_test)
    experiments.append(res2)

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment 3: Tuned LightGBM + Class Weights
    # ─────────────────────────────────────────────────────────────────────────
    print("Running Experiment 3: Tuned LightGBM + Class Weights...")
    model_exp3 = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multiclass",
        num_class=NUM_CLASSES,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        force_col_wise=True
    )
    res3 = train_and_eval("Tuned + Class Weights", model_exp3, X_train, y_train, X_val, y_val, X_test, y_test)
    experiments.append(res3)

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment 4: Controlled Random Oversampling + LightGBM
    # ─────────────────────────────────────────────────────────────────────────
    if RandomOverSampler is not None:
        print("Running Experiment 4: Controlled Random Oversampling...")
        ros = RandomOverSampler(random_state=42)
        X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
        model_exp4 = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            num_leaves=64,
            objective="multiclass",
            num_class=NUM_CLASSES,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True
        )
        res4 = train_and_eval("Oversampling", model_exp4, X_train_ros, y_train_ros, X_val, y_val, X_test, y_test)
        experiments.append(res4)

    # ─────────────────────────────────────────────────────────────────────────
    # Experiment 5: SMOTE + LightGBM
    # ─────────────────────────────────────────────────────────────────────────
    if SMOTE is not None:
        print("Running Experiment 5: SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        model_exp5 = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            num_leaves=64,
            objective="multiclass",
            num_class=NUM_CLASSES,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True
        )
        res5 = train_and_eval("SMOTE", model_exp5, X_train_smote, y_train_smote, X_val, y_val, X_test, y_test)
        experiments.append(res5)

    # ─────────────────────────────────────────────────────────────────────────
    # Model Selection based on Validation Macro F1
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("EXPERIMENT BENCHMARK SUMMARY")
    print("=" * 80)
    summary_cols = [
        "Experiment", "Val Accuracy", "Val Macro F1", "Val Weighted F1",
        "Test Accuracy", "Test Macro F1", "Test Weighted F1"
    ]
    summary_df = pd.DataFrame(experiments)[summary_cols]
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Select best model by Validation Macro F1
    best_exp = max(experiments, key=lambda x: x["Val Macro F1"])
    best_model = best_exp["_model"]
    print(f"\n>>> Selected Best Configuration: '{best_exp['Experiment']}' (Val Macro F1 = {best_exp['Val Macro F1']:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # Final Untouched Test Set Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    test_preds = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_macro_p = precision_score(y_test, test_preds, average="macro", zero_division=0)
    test_macro_r = recall_score(y_test, test_preds, average="macro", zero_division=0)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_weighted_p = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    test_weighted_r = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    test_bal_acc = balanced_accuracy_score(y_test, test_preds)

    print("\n" + "=" * 80)
    print(f"FINAL TEST SET EVALUATION ({best_exp['Experiment']})")
    print("=" * 80)
    print(f"  Test Accuracy:        {test_acc:.4f}")
    print(f"  Test Balanced Acc:    {test_bal_acc:.4f}")
    print(f"  Test Macro Precision: {test_macro_p:.4f}")
    print(f"  Test Macro Recall:    {test_macro_r:.4f}")
    print(f"  Test Macro F1:        {test_macro_f1:.4f}")
    print(f"  Test Weighted Prec:   {test_weighted_p:.4f}")
    print(f"  Test Weighted Recall: {test_weighted_r:.4f}")
    print(f"  Test Weighted F1:     {test_weighted_f1:.4f}")
    print()

    print("Classification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(y_test, test_preds)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=[f"True_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES]))

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Importance (Gain & Split)
    # ─────────────────────────────────────────────────────────────────────────
    booster = best_model.booster_
    importance_gain = booster.feature_importance(importance_type="gain")
    importance_split = booster.feature_importance(importance_type="split")

    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importance_gain,
        "importance_split": importance_split
    })

    print("\nTop 20 Features by Information Gain:")
    top_gain = feat_imp_df.sort_values(by="importance_gain", ascending=False).head(20)
    print(top_gain[["feature", "importance_gain"]].to_string(index=False))

    print("\nTop 20 Features by Split Count:")
    top_split = feat_imp_df.sort_values(by="importance_split", ascending=False).head(20)
    print(top_split[["feature", "importance_split"]].to_string(index=False))

    # ─────────────────────────────────────────────────────────────────────────
    # Save Artefacts
    # ─────────────────────────────────────────────────────────────────────────
    print("\nSaving artefacts...")

    # 1. Model artefacts
    model_v2_path = models_dir / "lgbm-v2.pkl"
    with open(model_v2_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Saved final model -> {model_v2_path}")

    # Also update lgbm-v1.pkl to ensure seamless pipeline compatibility
    model_v1_path = models_dir / "lgbm-v1.pkl"
    with open(model_v1_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Updated production baseline model -> {model_v1_path}")

    # 2. Class weights JSON
    cw_path = models_dir / "class_weights.json"
    with open(cw_path, "w") as f:
        json.dump({CLASS_NAMES[k]: v for k, v in cw_dict.items()}, f, indent=2)
    print(f"  Saved class weights -> {cw_path}")

    # 3. Feature names JSON
    feat_path = models_dir / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump({"feature_columns": feature_cols, "count": len(feature_cols)}, f, indent=2)
    print(f"  Saved feature names -> {feat_path}")

    # 4. Label Encoder PKL
    label_enc_path = models_dir / "label_encoder.pkl"
    with open(label_enc_path, "wb") as f:
        pickle.dump({"class_to_int": CLASS_TO_INT, "class_names": CLASS_NAMES}, f)
    print(f"  Saved label mapping -> {label_enc_path}")

    # 5. Experiment Results CSV
    csv_results = []
    for exp in experiments:
        row = {"dataset_version": "class_conditional_synthetic_v1"}
        row.update({k: v for k, v in exp.items() if not k.startswith("_")})
        csv_results.append(row)
        
    exp_csv_path = models_dir / "experiment_results.csv"
    new_results_df = pd.DataFrame(csv_results)
    
    if exp_csv_path.exists():
        try:
            old_df = pd.read_csv(exp_csv_path)
            if "dataset_version" not in old_df.columns:
                old_df.insert(0, "dataset_version", "synthetic_noise_baseline")
            combined_df = pd.concat([old_df, new_results_df], ignore_index=True)
            combined_df.to_csv(exp_csv_path, index=False)
        except Exception:
            new_results_df.to_csv(exp_csv_path, index=False)
    else:
        new_results_df.to_csv(exp_csv_path, index=False)
        
    print(f"  Saved experiment results -> {exp_csv_path}")

    print("\nAll training, evaluation, and persistence tasks completed successfully.")


if __name__ == "__main__":
    main()
