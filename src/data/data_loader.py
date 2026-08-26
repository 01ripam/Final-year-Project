"""
data_loader.py
──────────────
Production-ready CICIDS2017 Data Ingestion & Preprocessing Pipeline.

Responsibilities
────────────────
  1. Ingest one or more raw CICIDS2017 CSV files (glob or explicit paths).
  2. Standardise column names (strip whitespace, lowercase).
  3. Clean data quality issues:
       - Infinite values  (→ NaN, then imputed or dropped)
       - NaN / missing values (per-column strategy — drop or median-fill)
       - Duplicate flow records (deduplicated on flow 5-tuple + timestamp)
  4. Apply the Unified Attack Label Mapper → 7-class taxonomy.
  5. Produce a reproducible stratified 60 / 20 / 20 split.
  6. Log rich class-distribution summaries at every stage.
  7. Persist clean artefacts to disk (Parquet + optional CSV).

Usage (CLI)
───────────
    python data_loader.py \
        --input-dir  /data/raw/CICIDS2017 \
        --output-dir /data/processed \
        --label-col  " Label" \
        --drop-inf \
        --log-level  INFO

Usage (library)
───────────────
    from src.data.data_loader import CICIDSDataLoader
    loader = CICIDSDataLoader(input_dir="/data/raw/CICIDS2017")
    train_df, val_df, test_df = loader.run()

Author : Person 1 — Data & Classical ML
Project: Enterprise NIDS — Week 1 Foundation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.label_mapper import (
    CLASS_TO_INT,
    UNIFIED_CLASSES,
    map_label,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger("nids.data_loader")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Canonical feature set after label column removal.
# All numeric — matches CICIDS2017 CICFlowMeter output order.
LABEL_COLUMN_ALIASES: list[str] = [
    "label", " label", "Label", " Label",
    "class", "Class", "attack_cat", "Attack",
]

# Columns that are meta-data / identifiers — excluded from feature matrix
NON_FEATURE_COLS: set[str] = {
    "flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
    "timestamp", "unified_label", "label_int",
}

RANDOM_STATE:  int   = 42
TRAIN_RATIO:   float = 0.60
VAL_RATIO:     float = 0.20   # of total; test = 1 − TRAIN_RATIO − VAL_RATIO
TEST_RATIO:    float = 0.20

DEDUP_SUBSET: list[str] = [
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "flow_duration",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helper Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _log_class_distribution(df: pd.DataFrame, stage: str) -> None:
    """Emit a formatted class-distribution table to the logger."""
    counts = df["unified_label"].value_counts()
    total  = len(df)
    lines  = [f"\n{'─'*55}", f"  Class Distribution — {stage}", f"{'─'*55}"]
    for cls in UNIFIED_CLASSES:
        n   = counts.get(cls, 0)
        pct = 100.0 * n / total if total > 0 else 0.0
        lines.append(f"  {cls:<18} {n:>10,}  ({pct:5.2f}%)")
    lines += [f"{'─'*55}", f"  TOTAL              {total:>10,}", f"{'─'*55}"]
    logger.info("\n".join(lines))


def _detect_label_column(df: pd.DataFrame) -> str:
    """Auto-detect the raw label column from common CICIDS2017 aliases."""
    for alias in LABEL_COLUMN_ALIASES:
        if alias in df.columns:
            return alias
    # Fall back to case-insensitive search
    lower_map = {c.strip().lower(): c for c in df.columns}
    for alias in LABEL_COLUMN_ALIASES:
        key = alias.strip().lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(
        f"Could not detect a label column. Columns found: {list(df.columns)[:10]}. "
        f"Specify --label-col explicitly."
    )


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and lowercase them."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Core Loader Class
# ──────────────────────────────────────────────────────────────────────────────

class CICIDSDataLoader:
    """
    Stateful pipeline object for ingesting and preprocessing CICIDS2017.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing ``*.csv`` files (searched recursively).
    input_files : list[str | Path], optional
        Explicit list of CSV paths; takes priority over ``input_dir``.
    label_col : str, optional
        Override auto-detection of the label column name.
    drop_inf : bool
        If True, rows with infinite values are dropped.
        If False, infinite values are replaced with NaN and then imputed.
    nan_strategy : 'drop' | 'median'
        Strategy for handling NaN values after inf processing.
    dedup : bool
        Whether to deduplicate flow records on the 5-tuple.
    output_dir : str | Path, optional
        Where to persist Parquet artefacts.
    log_level : str
        Python logging level string.
    """

    def __init__(
        self,
        input_dir:   Optional[Union[str, Path]] = None,
        input_files: Optional[List[Union[str, Path]]] = None,
        label_col:   Optional[str] = None,
        drop_inf:    bool = True,
        nan_strategy: str = "median",   # 'drop' | 'median'
        dedup:       bool = True,
        output_dir:  Optional[Union[str, Path]] = None,
        log_level:   str = "INFO",
    ) -> None:
        logger.setLevel(log_level.upper())

        if input_dir is None and input_files is None:
            raise ValueError("Provide either `input_dir` or `input_files`.")

        self.input_dir    = Path(input_dir)   if input_dir   else None
        self.input_files  = [Path(f) for f in input_files] if input_files else []
        self.label_col    = label_col
        self.drop_inf     = drop_inf
        self.nan_strategy = nan_strategy.lower()
        self.dedup        = dedup
        self.output_dir   = Path(output_dir) if output_dir else None

        if self.nan_strategy not in {"drop", "median"}:
            raise ValueError("`nan_strategy` must be 'drop' or 'median'.")

        # Pipeline state — populated during run()
        self.raw_df:       Optional[pd.DataFrame] = None
        self.clean_df:     Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []

    # ── Step 1: Ingestion ────────────────────────────────────────────────────

    def _collect_csv_paths(self) -> List[Path]:
        paths: list[Path] = list(self.input_files)
        if self.input_dir:
            found = sorted(self.input_dir.rglob("*.csv"))
            if not found:
                raise FileNotFoundError(
                    f"No CSV files found under {self.input_dir}."
                )
            paths += found
            logger.info("Found %d CSV file(s) under %s.", len(found), self.input_dir)
        return paths

    def _ingest(self, paths: List[Path]) -> pd.DataFrame:
        """Read and concatenate all CSV files into a single DataFrame."""
        frames = []
        for p in paths:
            logger.info("Reading %s …", p.name)
            try:
                df = pd.read_csv(
                    p,
                    low_memory=False,
                    encoding="utf-8",
                    on_bad_lines="warn",
                )
                logger.info("  └─ %s: %d rows × %d cols", p.name, *df.shape)
                frames.append(df)
            except Exception as exc:
                logger.error("Failed to read %s: %s", p, exc)
                raise

        combined = pd.concat(frames, ignore_index=True)
        logger.info("Combined shape: %d rows × %d cols.", *combined.shape)
        return combined

    # ── Step 2: Standardise Columns ──────────────────────────────────────────

    def _standardise(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        df = _standardise_columns(df)
        # Re-detect label column after standardisation
        label_col = self.label_col.strip().lower().replace(" ", "_") if self.label_col else None
        if label_col is None or label_col not in df.columns:
            label_col = _detect_label_column(df)
            # Standardise its name too
            label_col = label_col.strip().lower().replace(" ", "_")
        logger.info("Label column detected: %r", label_col)
        return df, label_col

    # ── Step 3: Data Quality Cleaning ────────────────────────────────────────

    def _clean(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        initial_rows = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 3a. Duplicate removal
        if self.dedup:
            dedup_cols_present = [c for c in DEDUP_SUBSET if c in df.columns]
            if dedup_cols_present:
                before = len(df)
                df = df.drop_duplicates(subset=dedup_cols_present, keep="first")
                logger.info(
                    "Deduplication: removed %d duplicate rows (%d → %d).",
                    before - len(df), before, len(df),
                )
            else:
                df = df.drop_duplicates()
                logger.info(
                    "Deduplication (full row): %d → %d rows.", initial_rows, len(df)
                )

        # 3b. Infinite value handling
        inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
        n_inf = inf_mask.sum()
        if n_inf > 0:
            if self.drop_inf:
                df = df[~inf_mask].reset_index(drop=True)
                logger.info("Dropped %d rows containing infinite values.", n_inf)
            else:
                df[numeric_cols] = df[numeric_cols].replace(
                    [np.inf, -np.inf], np.nan
                )
                logger.info(
                    "Replaced %d infinite values with NaN (will be imputed).", n_inf
                )

        # 3c. NaN handling
        n_nan_rows = df[numeric_cols].isna().any(axis=1).sum()
        if n_nan_rows > 0:
            if self.nan_strategy == "drop":
                df = df.dropna(subset=numeric_cols).reset_index(drop=True)
                logger.info("Dropped %d rows with NaN values.", n_nan_rows)
            else:  # median imputation
                medians = df[numeric_cols].median()
                df[numeric_cols] = df[numeric_cols].fillna(medians)
                logger.info(
                    "Imputed NaN values in %d rows using per-column medians.", n_nan_rows
                )

        # 3d. Constant / zero-variance columns (log only — not dropped here)
        std_vals = df[numeric_cols].std()
        zero_var = std_vals[std_vals == 0].index.tolist()
        if zero_var:
            logger.warning(
                "Zero-variance columns detected (consider dropping): %s", zero_var
            )

        logger.info(
            "Cleaning complete. Rows: %d → %d (removed %d).",
            initial_rows, len(df), initial_rows - len(df),
        )
        return df

    # ── Step 4: Label Mapping ────────────────────────────────────────────────

    def _apply_label_mapping(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        logger.info("Applying unified attack label mapping …")
        df["unified_label"] = df[label_col].apply(
            lambda x: map_label(str(x), strict=False)
        )
        df["label_int"] = df["unified_label"].map(CLASS_TO_INT)
        _log_class_distribution(df, "Post-Mapping (Full Dataset)")
        return df

    # ── Step 5: Feature Column Selection ────────────────────────────────────

    def _select_features(self, df: pd.DataFrame, label_col: str) -> List[str]:
        exclude = NON_FEATURE_COLS | {label_col, "unified_label", "label_int"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        logger.info("Feature columns selected: %d", len(feature_cols))
        return feature_cols

    # ── Step 6: Stratified Split ─────────────────────────────────────────────

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Reproducible 60 / 20 / 20 stratified split.

        Strategy:
          Split 1: (80% train+val) vs (20% test)
          Split 2: (75% of 80% = 60% train) vs (25% of 80% = 20% val)
        """
        y = df["unified_label"]

        df_trainval, df_test = train_test_split(
            df,
            test_size=TEST_RATIO,
            stratify=y,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        val_ratio_of_trainval = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        df_train, df_val = train_test_split(
            df_trainval,
            test_size=val_ratio_of_trainval,
            stratify=df_trainval["unified_label"],
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        logger.info(
            "Split sizes — Train: %d | Val: %d | Test: %d",
            len(df_train), len(df_val), len(df_test),
        )
        _log_class_distribution(df_train, "TRAIN split")
        _log_class_distribution(df_val,   "VALIDATION split")
        _log_class_distribution(df_test,  "TEST split")

        return (
            df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True),
        )

    # ── Step 7: Summary Statistics ───────────────────────────────────────────

    def _log_summary_stats(self, df: pd.DataFrame) -> None:
        numeric_df = df[self.feature_cols]
        stats = numeric_df.describe().T[["mean", "std", "min", "max"]]
        logger.info("\nFeature Summary Statistics (first 10 features):\n%s", stats.head(10))

    # ── Step 8: Persistence ──────────────────────────────────────────────────

    def _persist(
        self,
        df_train: pd.DataFrame,
        df_val:   pd.DataFrame,
        df_test:  pd.DataFrame,
    ) -> None:
        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        splits = {"train": df_train, "val": df_val, "test": df_test}
        for name, df in splits.items():
            out = self.output_dir / f"{name}.parquet"
            df.to_parquet(out, index=False, compression="snappy")
            logger.info("Saved %s → %s (%d rows)", name, out, len(df))

        # Write feature column list for downstream reproducibility
        feat_path = self.output_dir / "feature_columns.txt"
        feat_path.write_text("\n".join(self.feature_cols))
        logger.info("Feature column list saved → %s", feat_path)

    # ── Public Entrypoint ────────────────────────────────────────────────────

    def run(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the full pipeline.

        Returns
        -------
        (train_df, val_df, test_df) : tuple of DataFrames
            Each DataFrame contains feature columns + 'unified_label' + 'label_int'.
        """
        # 1. Collect paths
        paths = self._collect_csv_paths()

        # 2. Ingest
        df = self._ingest(paths)
        self.raw_df = df.copy()

        # 3. Standardise
        df, label_col = self._standardise(df)

        # 4. Clean
        df = self._clean(df, label_col)

        # 5. Map labels
        df = self._apply_label_mapping(df, label_col)

        # 6. Select features
        self.feature_cols = self._select_features(df, label_col)

        # 7. Summary stats
        self.clean_df = df
        self._log_summary_stats(df)

        # 8. Split
        df_train, df_val, df_test = self._split(df)

        # 9. Persist
        self._persist(df_train, df_val, df_test)

        logger.info("Pipeline complete. Ready for modelling.")
        return df_train, df_val, df_test


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data_loader",
        description="CICIDS2017 data ingestion & preprocessing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", type=str, default=None,
        help="Directory containing raw CICIDS2017 CSV files.",
    )
    p.add_argument(
        "--input-files", nargs="+", type=str, default=None,
        help="Explicit CSV file paths (space-separated).",
    )
    p.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for Parquet artefacts.",
    )
    p.add_argument(
        "--label-col", type=str, default=None,
        help="Override auto-detection of the label column name.",
    )
    p.add_argument(
        "--drop-inf", action="store_true", default=True,
        help="Drop rows containing inf values (default). Use --no-drop-inf to impute.",
    )
    p.add_argument("--no-drop-inf", dest="drop_inf", action="store_false")
    p.add_argument(
        "--nan-strategy", choices=["drop", "median"], default="median",
        help="NaN handling strategy.",
    )
    p.add_argument(
        "--no-dedup", dest="dedup", action="store_false", default=True,
        help="Disable duplicate flow removal.",
    )
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    loader = CICIDSDataLoader(
        input_dir    = args.input_dir,
        input_files  = args.input_files,
        label_col    = args.label_col,
        drop_inf     = args.drop_inf,
        nan_strategy = args.nan_strategy,
        dedup        = args.dedup,
        output_dir   = args.output_dir,
        log_level    = args.log_level,
    )
    train_df, val_df, test_df = loader.run()

    print(f"\nDone. Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")


if __name__ == "__main__":
    main()
