-- ─────────────────────────────────────────────────────────────────────────────
-- Enterprise NIDS — TimescaleDB Schema Initialisation
-- Runs once on first container start via docker-entrypoint-initdb.d/
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ── Flows Table (raw feature vectors) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS flows (
    flow_id             UUID         NOT NULL,
    capture_timestamp   TIMESTAMPTZ  NOT NULL,
    src_ip              INET         NOT NULL,
    dst_ip              INET         NOT NULL,
    src_port            SMALLINT     NOT NULL,
    dst_port            SMALLINT     NOT NULL,
    protocol            SMALLINT     NOT NULL,
    flow_duration       DOUBLE PRECISION,
    total_fwd_packets   INTEGER,
    total_bwd_packets   INTEGER,
    sensor_id           TEXT,
    network_segment     TEXT,
    -- Feature payload stored as JSONB for flexibility during schema evolution
    features            JSONB        NOT NULL,
    PRIMARY KEY (flow_id, capture_timestamp)
);

-- Convert to hypertable partitioned on capture_timestamp (7-day chunks)
SELECT create_hypertable(
    'flows', 'capture_timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

-- ── Predictions Table ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    flow_id                UUID         NOT NULL,
    inference_timestamp    TIMESTAMPTZ  NOT NULL,
    predicted_class        TEXT         NOT NULL,
    confidence             REAL         NOT NULL,
    class_probabilities    JSONB,
    model_name             TEXT,
    inference_latency_ms   REAL,
    shap_queued            BOOLEAN      DEFAULT FALSE,
    PRIMARY KEY (flow_id, inference_timestamp),
    FOREIGN KEY (flow_id, inference_timestamp)
        REFERENCES flows(flow_id, capture_timestamp)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

SELECT create_hypertable(
    'predictions', 'inference_timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

-- ── SHAP Explanations Table ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shap_explanations (
    flow_id                  UUID         NOT NULL,
    explanation_timestamp    TIMESTAMPTZ  NOT NULL,
    shap_values              JSONB        NOT NULL,
    base_value               REAL,
    top_features             TEXT[],
    explainer_type           TEXT,
    explanation_latency_ms   REAL,
    PRIMARY KEY (flow_id, explanation_timestamp)
);

SELECT create_hypertable(
    'shap_explanations', 'explanation_timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

-- ── Useful Indexes ────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_flows_src_ip    ON flows (src_ip, capture_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_flows_dst_ip    ON flows (dst_ip, capture_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_preds_class     ON predictions (predicted_class, inference_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_preds_alert     ON predictions (predicted_class)
    WHERE predicted_class != 'Normal';

-- ── Continuous Aggregate: Attack alerts per minute ────────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS alerts_per_minute
    WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', inference_timestamp) AS bucket,
    predicted_class,
    COUNT(*)                                      AS alert_count,
    AVG(confidence)                               AS avg_confidence
FROM predictions
WHERE predicted_class != 'Normal'
GROUP BY bucket, predicted_class
WITH NO DATA;

SELECT add_continuous_aggregate_policy('alerts_per_minute',
    start_offset => INTERVAL '1 hour',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);
