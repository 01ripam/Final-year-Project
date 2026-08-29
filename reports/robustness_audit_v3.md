# Robustness Audit – Version 3

## Overview
This document provides a concise audit of the current **Enterprise Network Intrusion Detection System (NIDS)** synthetic data generation and training pipeline. The audit precedes any code modifications and establishes a baseline for the next robustness and production‑readiness improvement phase.

## Current Implementation Summary
| Component | File | Key Responsibilities |
|-----------|------|----------------------|
| Synthetic Generator | [generate_synthetic.py](file:///C:/Users/91906/Desktop/Final-year-Project/src/data/generate_synthetic.py) | Generates 78 canonical numeric features conditioned on the 7‑class taxonomy, supports two profiles (`v1` standard, `v2` shifted OOD). Handles label distribution, flow identity fields, feature calculation, and validation. |
| Label Mapping | [label_mapper.py](file:///C:/Users/91906/Desktop/Final-year-Project/src/data/label_mapper.py) | Defines `UNIFIED_CLASSES` (the 7‑class taxonomy) and provides utilities for mapping raw CICIDS2017 labels to the unified set. |
| Data Contract | [flow_schema.py](file:///C:/Users/91906/Desktop/Final-year-Project/src/data/flow_schema.py) | Pydantic V2 model (`FlowRecord`) that enforces the 78‑feature contract, includes serialization helpers (`to_redis_payload`, `to_feature_array`). |
| Training Pipeline | [train_lgbm.py](file:///C:/Users/91906/Desktop/Final-year-Project/src/train/train_lgbm.py) | Loads data, performs LightGBM training with optional class‑balancing, hyper‑parameter tuning, and SMOTE oversampling. Reports baseline, balanced, tuned, and oversampled metrics. |
| Randomness Control | `np.random.default_rng(seed)` in the generator (line 684) | Guarantees reproducible synthetic datasets when the same seed is supplied. |

## Strengths
- **Feature Contract Stability** – The generator always emits the 78‑feature set defined in `DEFAULT_FEATURE_COLUMNS`; validation (`validate_generated_dataset`) checks for missing columns, NaNs, Infs, and negative values.
- **Reproducibility** – Fixed seed handling via `np.random.default_rng` guarantees deterministic output for a given seed and profile.
- **Multi‑Modal Profiles** – Two configurable profiles (`v1`, `v2`) enable out‑of‑distribution testing without code changes.
- **Extensive Validation** – Checks for column presence, NaNs/Infs, and non‑negative core metrics before persisting data.
- **Modular Design** – Separate functions for label generation, flow identities, feature generation, and validation make unit testing straightforward (all 126 unit tests currently pass).

## Weaknesses / Risks
1. **Hard‑Coded Profile Logic** – Profile branching is embedded directly in `generate_class_labels` and `generate_class_features`. Adding new profiles requires code changes, increasing maintenance burden.
2. **Limited Perturbation Hooks** – Stress‑testing or feature‑level perturbations must be injected into the generator logic; there is no explicit abstraction for post‑generation transformation.
3. **No Data Versioning** – Synthetic outputs are written directly to CSV without any version metadata (e.g., seed, profile, generation timestamp) embedded in the file.
4. **Validation Scope** – `validate_generated_dataset` checks for NaNs/Infs and a few non‑negative columns, but does not verify statistical properties (e.g., feature distributions, inter‑class overlap) that are crucial for robustness.
5. **Static Feature Columns** – `DEFAULT_FEATURE_COLUMNS` is imported from `src.models.dummy`; any change to the canonical list would require coordinated updates across generator, schema, and training scripts.
6. **Absence of Runtime Configuration** – The generator CLI only accepts command‑line arguments; there is no programmatic API (e.g., a class) for downstream pipelines to request data with custom perturbations.
7. **Potential Data Leakage** – The generator mixes identity fields (`Flow ID`, IPs, ports) with feature values but does not provide a clear separation for privacy‑preserving pipelines.

## Recommendations (Immediate)
- **Wrap Generator in a Class** – Create a `SyntheticDataGenerator` class exposing methods like `generate(labels, profile)`, `apply_perturbations`, and `export`. This will simplify adding new profiles and stress‑testing hooks.
- **Add Metadata Column** – Include columns such as `generation_timestamp`, `seed`, `profile` to aid reproducibility tracking and experiment logging.
- **Expand Validation** – Implement statistical sanity checks (e.g., Kolmogorov‑Smirnov test against reference distributions) and inter‑class overlap metrics.
- **Expose a Config Object** – Use a Pydantic config model for generator parameters (noise rate, shifts, jitter) to enable external configuration without code edits.
- **Separate Identity from Features** – Optionally produce two outputs: a pure feature matrix and a metadata sheet, allowing downstream privacy‑aware pipelines to drop identity fields safely.
- **Unit‑Test Perturbation Paths** – Add tests that verify the correctness of any future perturbation functions (e.g., multiplicative scaling, Gaussian noise).

## Next Steps (Phase 2 – Shortcut Analysis)
1. Implement `src/analysis/feature_stats.py` to compute per‑class feature statistics, correlation matrices, and importance scores.
2. Use the new analysis module to detect features with negligible variance or high collinearity – candidates for **feature ablation**.
3. Document findings in `reports/feature_stats_v1.md` to inform subsequent robustness experiments.

---
*Prepared on* `2026-08-29` *by Antigravity – adhering to the 78‑feature contract and reproducibility constraints.*
