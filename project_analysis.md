# Enterprise NIDS — Final Year Project Analysis

## What This Project Is

An **Enterprise Network Intrusion Detection System (NIDS)** — a full-stack ML-powered security platform that:
1. Captures live network packets
2. Aggregates them into **network flows** (TCP/UDP sessions)
3. Extracts **78 CICIDS2017-compatible features** per flow
4. Classifies flows into **7 attack categories** in real-time using ML/DL models
5. Generates **SHAP explanations** for every malicious prediction
6. Stores everything in **TimescaleDB** (time-series PostgreSQL)
7. Visualises alerts on a **real-time SOC dashboard** (React + WebSocket)

### 7-Class Attack Taxonomy
| Int | Class | Examples |
|-----|-------|---------|
| 0 | Normal | Benign traffic |
| 1 | DoS | Hulk, GoldenEye, Slowloris, Heartbleed |
| 2 | DDoS | LOIC |
| 3 | Port Scan | PortScan, Infiltration |
| 4 | Brute Force | FTP-Patator, SSH-Patator |
| 5 | Botnet | ARES |
| 6 | Web Attack | XSS, SQLi, Brute Force |

---

## Architecture Overview

```
Live Network Traffic
       │
       ▼
[Network Engine]  — Scapy/PyShark packet capture
       │  FlowFeatureVector (78 features, Pydantic)
       ▼
[Redis Queue]  — flow:queue (LPUSH / BRPOP)
       │
       ▼
[FastAPI ML Inference]  — REST + WebSocket
   ├── [Celery Inference Worker]  — LightGBM / XGBoost / PyTorch
   ├── [SHAP Worker]              — TreeExplainer / DeepExplainer
   └── [DB Writer Worker]         — asyncpg → TimescaleDB
                                        │
                                        ▼
                              [TimescaleDB Hypertables]
                                 flows / predictions / shap_explanations
                                        │
                               [Grafana Dashboards]
                                        │
                               [React SOC Frontend]  ← WebSocket alerts
```

---

## What's DONE ✅

### Infrastructure (Week 1 — Complete)
- [x] `docker-compose.yml` — Redis, TimescaleDB, Grafana fully wired
- [x] `infra/sql/init.sql` — 3 hypertables (`flows`, `predictions`, `shap_explanations`), continuous aggregate, indexes
- [x] `requirements.txt` — All dependencies pinned

### Data Layer (Week 1 — Complete)
- [x] `src/data/label_mapper.py` — Full CICIDS2017 → 7-class mapper with fuzzy matching, strict mode, coverage audit
- [x] `src/data/flow_schema.py` — Pydantic V2 schema: `FlowFeatureVector` (78 fields), `FlowPrediction`, `ShapExplanation` with Redis/DB serialisers
- [x] `src/data/data_loader.py` — Full CICIDS2017 ingestion pipeline: CSV → clean → label-map → 60/20/20 stratified split → Parquet

### Tests (Week 1 — Complete)
- [x] `tests/unit/test_label_mapper.py` — 10 tests, all CICIDS2017 variants
- [x] `tests/unit/test_flow_schema.py` — 13 tests, validation + serialisation round-trips

---

## What's TODO ❌

### 🔴 HIGH PRIORITY — Core Pipeline (Week 2)

#### Person 3 — Network Engine
- [ ] `src/network_engine/packet_capture.py`
  - Scapy/PyShark live packet capture loop
  - Flow aggregation (5-tuple keying, timeout-based finalisation)
  - Push completed `FlowFeatureVector` to Redis `flow:queue`
- [ ] `src/network_engine/feature_extractor.py`
  - Compute all 78 CICIDS2017 features from raw packet list
  - Directional stats (fwd/bwd), IAT, TCP flags, bulk metrics, active/idle time

#### Person 1 + 2 — ML Workers
- [ ] `src/workers/inference_worker.py`
  - Celery worker consuming from Redis `flow:queue`
  - Load LightGBM / XGBoost (Person 1) and PyTorch model (Person 2)
  - Batch inference → produce `FlowPrediction`
  - Enqueue malicious flows to SHAP queue

#### Person 2 — XAI Worker
- [ ] `src/workers/shap_worker.py`
  - Celery worker consuming SHAP queue
  - TreeExplainer (LightGBM/XGB) or DeepExplainer (PyTorch)
  - Produce `ShapExplanation` → DB

#### Person 4 — Platform
- [ ] `src/workers/db_writer.py`
  - asyncpg bulk-write `FlowFeatureVector` + `FlowPrediction` + `ShapExplanation` to TimescaleDB
- [ ] `src/api/main.py`
  - FastAPI app: `/predict` REST endpoint + WebSocket `/ws/alerts`
  - Integrate Prometheus metrics instrumentation
  - Health check endpoints

#### Person 1 — ML Model Training
- [ ] Train LightGBM / XGBoost on processed CICIDS2017 Parquet files
- [ ] Save model artefacts (`.pkl` / `.json`)
- [ ] Evaluate: accuracy, F1-macro, confusion matrix

#### Person 2 — Deep Learning
- [ ] Design and train PyTorch model (MLP / 1D-CNN / TabTransformer)
- [ ] Save model weights (`.pt`)
- [ ] Integrate SHAP `DeepExplainer` / `GradientExplainer`

### 🟡 MEDIUM PRIORITY — Frontend & Observability

#### React SOC Dashboard (stub — commented out in docker-compose)
- [ ] Create `frontend/` directory with React app
- [ ] Real-time alert feed (WebSocket)
- [ ] Attack class distribution charts
- [ ] SHAP feature importance panel
- [ ] Flow inspector (click an alert → see all 78 features + SHAP waterfall)
- [ ] Wire up `frontend` service in `docker-compose.yml`

#### Grafana
- [ ] `infra/grafana/provisioning/` — datasource YAML (TimescaleDB connection)
- [ ] Dashboard JSON: alerts_per_minute, top attack classes, model latency

### 🟢 LOW PRIORITY — Polish & DevOps

- [ ] `.env.example` file — document env vars (POSTGRES_USER, POSTGRES_PASSWORD, GRAFANA_PASSWORD)
- [ ] `Dockerfile` for FastAPI + Celery workers (currently commented out in compose)
- [ ] `README.md` — setup, architecture, how to run
- [ ] Integration tests (`tests/integration/`)
- [ ] `pytest.ini` / `pyproject.toml` for test configuration
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] `src/models/` — currently empty (model artefacts + loader module needed)
- [ ] Fix relative import in `data_loader.py` (uses `from label_mapper import ...` — should be `from src.data.label_mapper import ...` when run as a library)

---

## Week-by-Week Roadmap

| Week | Focus | Status |
|------|-------|--------|
| 1 | Infra + Data Foundation | ✅ Done |
| 2 | Network Engine + ML Workers + FastAPI | 🔴 In Progress |
| 3 | Frontend + Grafana + Model Training | ⬜ Not Started |
| 4 | Integration Testing + Documentation + Demo | ⬜ Not Started |

---

## Quick Start (What Works Today)

```bash
# 1. Spin up infrastructure
docker compose up -d

# 2. Run unit tests
pytest tests/unit/ -v

# 3. Run data pipeline on CICIDS2017 CSVs
python src/data/data_loader.py \
  --input-dir /path/to/CICIDS2017 \
  --output-dir data/processed
```
