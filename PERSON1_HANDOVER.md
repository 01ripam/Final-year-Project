# PERSON 1 HANDOVER DOCUMENT

## 1. What Person 1 completed
I have successfully implemented the core Machine Learning training and inference pipeline. I resolved a critical feature mismatch issue (76 vs 78 features), implemented the `LightGBMClassifier` wrapper, configured the Celery `inference_worker.py` to use the production model instead of the dummy model, and ensured everything is unit-tested and SHAP-compatible.

## 2. Final ML architecture
- **Model**: LightGBM Classifier (Tree-based model, excellent for tabular NIDS data).
- **Registry**: `ModelRegistry` singleton loads models dynamically on worker boot.
- **Inference**: Celery `run_inference` task consumes from Redis `flow:queue`, extracts exactly 78 features via the Pydantic schema, and runs `predict_proba`.
- **Explainability**: Malicious predictions trigger a secondary asynchronous `shap_worker` task for TreeSHAP analysis.

## 3. Dataset information
Processed Parquet data is expected in `data/processed/`. The `generate_synthetic.py` script was used for my testing. To train a production model, use the real CICIDS2017 dataset with `data_loader.py`.

## 4. Canonical 78 features
The canonical feature list is strictly defined in `data/processed/feature_columns.txt`.
- `source_port` is excluded.
- `destination_port` and `protocol` are included.
The inference schema (`FlowFeatureVector`) maps `dst_port` to `destination_port` safely.

## 5. Label mapping
Classes are mapped dynamically via `src/data/label_mapper.py` into exactly 7 classes: `Normal`, `DoS`, `DDoS`, `Port Scan`, `Brute Force`, `Botnet`, `Web Attack`.

## 6. Final LightGBM model
The model uses `class_weight="balanced"` to combat the extreme class imbalances found in NIDS datasets, and trains using early stopping on the validation set.

## 7. Model file location
The trained artifact is saved at `models/lgbm-v1.pkl`.

## 8. How to retrain
```powershell
.venv\Scripts\python.exe src/train/train_lgbm.py
```

## 9. How to evaluate
Evaluation is built directly into the training script and runs automatically against the unseen `test.parquet` dataset at the end of training.

## 10. How inference works
The `inference_worker` runs continuously, blocking on a Redis `BRPOP`. When a flow arrives, it extracts the 78 canonical features (enforcing order) and calls `model.predict()`. It publishes the result to TimescaleDB and Redis pub/sub.

## 11. How ModelRegistry works
The model wrapper implements `BaseClassifier`. On worker startup, `inference_worker.py` calls `_ensure_model_registered()`, which initializes `LightGBMClassifier("models/lgbm-v1.pkl")` and registers it as the active model.

## 12. How Celery/Redis connects to inference
Redis provides the message queues (`flow:queue` and pub/sub channels). Celery provides the async task workers that pull from Redis and execute the Python code.

## 13. How SHAP works
The `shap_worker.py` consumes malicious `FlowPrediction` events. It uses `shap.TreeExplainer` on the native LightGBM `booster_` to calculate feature contributions, which it then publishes to Redis for the UI.

## 14. Test results
Run `pytest tests/unit -v`. All 126 unit tests are passing.

## 15. Actual model metrics
*(Based on the synthetic data run)*
- Accuracy: ~66%
- Macro F1: ~13%
- Note: These metrics reflect the synthetic Gaussian noise dataset used for testing the pipeline, not actual network signal.

## 16. Commands to reproduce the ML pipeline
```powershell
.venv\Scripts\python.exe src/train/train_lgbm.py
.venv\Scripts\python.exe src/train/train_pytorch.py
.venv\Scripts\pytest.exe tests/unit -v
```

## 17. Environment requirements
Python 3.12, Docker (for Redis and TimescaleDB). Dependencies are in `requirements.txt`.

## 18. What Person 2/other team members still need to implement
- **Person 2 (Deep Learning)**: Tune and improve the `pytorch.py` and `train_pytorch.py` implementation if needed. I updated it to 78 features so it doesn't break, but the architecture and tuning is your domain.
- **Backend/Frontend Team**: Implement the WebSocket server that listens to the Redis `alerts` and `shap_explanations` channels and streams them to the React dashboard. Connect the live packet capture `feature_extractor` to the Redis `flow:queue`.

## 19. Known limitations
The current model is trained on synthetic data. Real CICIDS2017 ingestion needs to be run before deployment.

## 20. Troubleshooting
If `inference_worker.py` fails with a KeyError on `source_port`, ensure you are using the updated `feature_columns.txt` (which excludes `source_port`) and that the model was trained with exactly 78 features.
