# Enterprise NIDS (Final Year Project)

An enterprise-grade Network Intrusion Detection System (NIDS) pipeline incorporating real-time packet capture, machine learning inference (LightGBM), deep learning (PyTorch), and explainable AI (SHAP) across a distributed Celery/Redis architecture.

## Person 1: ML Architecture & Inference

This repository contains the completed baseline ML architecture. The system uses a LightGBM classifier deployed via a Celery worker for real-time inference on 78 canonical network flow features.

### Python Environment
- Python 3.12
- Uses `.venv` with dependencies defined in `requirements.txt`.
- Install dependencies: `pip install -r requirements.txt`

### Dataset Information
- The system expects CICIDS2017-compatible parquet files in `data/processed/`.
- Training data: `data/processed/train.parquet`
- Validation data: `data/processed/val.parquet`
- Test data: `data/processed/test.parquet`

### The 78-Feature Contract
The pipeline enforces a strict 78-feature contract based on `data/processed/feature_columns.txt`. 
The `source_port` is excluded intentionally to prevent overfitting to ephemeral ports, while `destination_port` and `protocol` are retained.

### 7-Class Label Mapping
The model maps raw network behaviors to 7 unified classes:
0. Normal
1. DoS
2. DDoS
3. Port Scan
4. Brute Force
5. Botnet
6. Web Attack

### Model Training & Evaluation
To retrain the LightGBM model on the processed dataset:
```bash
python src/train/train_lgbm.py
```
This script automatically applies class weighting, performs early stopping against the validation set, and outputs the final metrics (Accuracy, F1, Confusion Matrix) based on the unseen test set.

### Artifacts
The finalized ML artifact is saved to:
`models/lgbm-v1.pkl`

### Inference Pipeline
1. **Network Flow**: Generated as JSON by the feature extractor.
2. **Redis**: Pushed to the `flow:queue`.
3. **Inference Worker**: The `inference_worker.py` Celery process pulls vectors, maps the JSON schema to the 78 canonical features (resolving `dst_port` to `destination_port`), and feeds it to LightGBM.
4. **Model Registry**: Models are loaded dynamically via `src/models/registry.py`.

### How to Run Services (Celery / Redis / Tests)
- Start Redis (requires Docker): `docker-compose up -d redis`
- Start Celery workers: 
  ```bash
  celery -A src.workers.celery_app worker --loglevel=info
  ```
- Run unit tests: 
  ```bash
  pytest tests/unit -v
  ```

### Explainable AI (SHAP)
Malicious predictions flagged by the inference worker are forwarded asynchronously to the SHAP worker (`src/workers/shap_worker.py`). The SHAP worker accesses the native `model.booster_` from the LightGBM wrapper to compute tree-based local explanations.

### Known Limitations
- The provided `generate_synthetic.py` produces Gaussian noise, meaning metrics currently appear as random guessing (F1 Macro ~12%). To achieve realistic metrics, the full CICIDS2017 CSVs must be ingested using `src/data/data_loader.py`.
- Redis pub/sub for SHAP results assumes a live WebSocket server is listening, which must be implemented by the backend team.
