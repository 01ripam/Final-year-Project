"""
flow_schema.py
──────────────
Redis Flow Vector Data Contract — Pydantic V2 Schema.

Defines the canonical JSON payload that travels through the pipeline:

    Network Engine (Person 3)
        └─► Redis Queue  ──► FastAPI Inference (Person 4)
                                └─► ML Workers  (Person 1 / 2)
                                └─► SHAP Worker (Person 2)
                                └─► TimescaleDB Writer

All 78 CICIDS2017-compatible features are declared with:
  - Precise type annotations
  - Documented field semantics
  - Conservative validator constraints (non-negative where physically required)
  - Serialisation helpers for Redis (JSON) and DB (dict) round-trips

Author : Cross-team data contract (owned by Platform — Person 4)
Project: Enterprise NIDS — Week 1 Foundation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class AttackClass(str, Enum):
    NORMAL      = "Normal"
    DOS         = "DoS"
    DDOS        = "DDoS"
    PORT_SCAN   = "Port Scan"
    BRUTE_FORCE = "Brute Force"
    BOTNET      = "Botnet"
    WEB_ATTACK  = "Web Attack"


class Protocol(int, Enum):
    TCP  = 6
    UDP  = 17
    ICMP = 1
    OTHER = 0


# ──────────────────────────────────────────────────────────────────────────────
# Flow Feature Vector  (78 features — CICIDS2017 compatible)
# ──────────────────────────────────────────────────────────────────────────────

class FlowFeatureVector(BaseModel):
    """
    Canonical representation of a completed network flow.

    Lifecycle
    ---------
    1. Created by the Network Engine after a time-window expires.
    2. Serialised to JSON and pushed to the Redis ``flow:queue`` list.
    3. Consumed by the FastAPI ML Inference worker.
    4. Prediction + confidence scores appended → ``FlowPrediction``.
    5. Malicious flows forwarded to the async SHAP queue.
    6. Both records written to TimescaleDB in bulk batches.
    7. Alert payload broadcast over WebSocket to the SOC dashboard.
    """

    model_config = {
        "populate_by_name": True,
        "use_enum_values": True,
        "json_schema_extra": {
            "title": "FlowFeatureVector",
            "description": "79-field network flow vector for ML inference.",
        },
    }

    # ── Flow Identity ────────────────────────────────────────────────────────
    flow_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique flow identifier (UUID4).",
    )
    capture_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the flow was finalised by the aggregator.",
    )
    src_ip: str = Field(..., description="Source IP address (dotted-decimal or IPv6).")
    dst_ip: str = Field(..., description="Destination IP address.")
    src_port: int = Field(..., ge=0, le=65535, description="Source TCP/UDP port.")
    dst_port: int = Field(..., ge=0, le=65535, description="Destination TCP/UDP port.")
    protocol: int = Field(..., description="IP protocol number (6=TCP, 17=UDP, 1=ICMP).")

    # ── Flow Duration ────────────────────────────────────────────────────────
    flow_duration: float = Field(
        ..., ge=0.0, description="Total flow duration in microseconds."
    )

    # ── Packet Counts ────────────────────────────────────────────────────────
    total_fwd_packets: int = Field(..., ge=0, description="Packets in forward direction.")
    total_bwd_packets: int = Field(..., ge=0, description="Packets in backward direction.")
    total_length_fwd_packets: float = Field(..., ge=0.0, description="Total fwd payload bytes.")
    total_length_bwd_packets: float = Field(..., ge=0.0, description="Total bwd payload bytes.")

    # ── Fwd Packet Length Stats ──────────────────────────────────────────────
    fwd_packet_length_max:  float = Field(..., ge=0.0)
    fwd_packet_length_min:  float = Field(..., ge=0.0)
    fwd_packet_length_mean: float = Field(..., ge=0.0)
    fwd_packet_length_std:  float = Field(..., ge=0.0)

    # ── Bwd Packet Length Stats ──────────────────────────────────────────────
    bwd_packet_length_max:  float = Field(..., ge=0.0)
    bwd_packet_length_min:  float = Field(..., ge=0.0)
    bwd_packet_length_mean: float = Field(..., ge=0.0)
    bwd_packet_length_std:  float = Field(..., ge=0.0)

    # ── Flow Byte / Packet Rates ─────────────────────────────────────────────
    flow_bytes_per_sec:   float = Field(..., ge=0.0, description="Bytes/s (both directions).")
    flow_packets_per_sec: float = Field(..., ge=0.0, description="Packets/s (both directions).")

    # ── Inter-Arrival Times (IAT) — Flow Level ───────────────────────────────
    flow_iat_mean: float = Field(..., ge=0.0)
    flow_iat_std:  float = Field(..., ge=0.0)
    flow_iat_max:  float = Field(..., ge=0.0)
    flow_iat_min:  float = Field(..., ge=0.0)

    # ── IAT — Forward Direction ──────────────────────────────────────────────
    fwd_iat_total: float = Field(..., ge=0.0)
    fwd_iat_mean:  float = Field(..., ge=0.0)
    fwd_iat_std:   float = Field(..., ge=0.0)
    fwd_iat_max:   float = Field(..., ge=0.0)
    fwd_iat_min:   float = Field(..., ge=0.0)

    # ── IAT — Backward Direction ─────────────────────────────────────────────
    bwd_iat_total: float = Field(..., ge=0.0)
    bwd_iat_mean:  float = Field(..., ge=0.0)
    bwd_iat_std:   float = Field(..., ge=0.0)
    bwd_iat_max:   float = Field(..., ge=0.0)
    bwd_iat_min:   float = Field(..., ge=0.0)

    # ── TCP Flag Counts (per direction) ─────────────────────────────────────
    fwd_psh_flags: int = Field(..., ge=0)
    bwd_psh_flags: int = Field(..., ge=0)
    fwd_urg_flags: int = Field(..., ge=0)
    bwd_urg_flags: int = Field(..., ge=0)
    fin_flag_count: int = Field(..., ge=0)
    syn_flag_count: int = Field(..., ge=0)
    rst_flag_count: int = Field(..., ge=0)
    psh_flag_count: int = Field(..., ge=0)
    ack_flag_count: int = Field(..., ge=0)
    urg_flag_count: int = Field(..., ge=0)
    cwe_flag_count: int = Field(..., ge=0)
    ece_flag_count: int = Field(..., ge=0)

    # ── Header Lengths ───────────────────────────────────────────────────────
    fwd_header_length:  int = Field(..., ge=0)
    bwd_header_length:  int = Field(..., ge=0)

    # ── Packet Rates (directional) ───────────────────────────────────────────
    fwd_packets_per_sec: float = Field(..., ge=0.0)
    bwd_packets_per_sec: float = Field(..., ge=0.0)

    # ── Payload / Segment Size Stats ─────────────────────────────────────────
    min_packet_length:     float = Field(..., ge=0.0)
    max_packet_length:     float = Field(..., ge=0.0)
    packet_length_mean:    float = Field(..., ge=0.0)
    packet_length_std:     float = Field(..., ge=0.0)
    packet_length_variance: float = Field(..., ge=0.0)

    # ── Active / Idle Time ───────────────────────────────────────────────────
    active_mean: float = Field(..., ge=0.0)
    active_std:  float = Field(..., ge=0.0)
    active_max:  float = Field(..., ge=0.0)
    active_min:  float = Field(..., ge=0.0)
    idle_mean:   float = Field(..., ge=0.0)
    idle_std:    float = Field(..., ge=0.0)
    idle_max:    float = Field(..., ge=0.0)
    idle_min:    float = Field(..., ge=0.0)

    # ── Sub-flow Metrics ─────────────────────────────────────────────────────
    subflow_fwd_packets: int   = Field(..., ge=0)
    subflow_fwd_bytes:   float = Field(..., ge=0.0)
    subflow_bwd_packets: int   = Field(..., ge=0)
    subflow_bwd_bytes:   float = Field(..., ge=0.0)

    # ── Window / Bulk Metrics ────────────────────────────────────────────────
    init_win_bytes_fwd: int = Field(..., ge=-1, description="-1 indicates not available (non-TCP).")
    init_win_bytes_bwd: int = Field(..., ge=-1)
    act_data_pkt_fwd:   int = Field(..., ge=0)
    min_seg_size_fwd:   int = Field(..., ge=0)

    # ── Derived Ratios ───────────────────────────────────────────────────────
    down_up_ratio: float = Field(
        ..., description="Ratio of download to upload packets. Can be 0 for one-directional flows."
    )
    avg_packet_size:   float = Field(..., ge=0.0)
    avg_fwd_segment_size: float = Field(..., ge=0.0)
    avg_bwd_segment_size: float = Field(..., ge=0.0)
    fwd_avg_bytes_per_bulk:   float = Field(..., ge=0.0)
    fwd_avg_packets_per_bulk: float = Field(..., ge=0.0)
    fwd_avg_bulk_rate:        float = Field(..., ge=0.0)
    bwd_avg_bytes_per_bulk:   float = Field(..., ge=0.0)
    bwd_avg_packets_per_bulk: float = Field(..., ge=0.0)
    bwd_avg_bulk_rate:        float = Field(..., ge=0.0)

    # ── Capture Metadata (not used as features — for routing/audit) ──────────
    sensor_id: Optional[str] = Field(
        default=None,
        description="Network sensor / interface identifier (e.g., 'eth0', 'sensor-dmz-01').",
    )
    network_segment: Optional[str] = Field(
        default=None,
        description="Logical network segment label (e.g., 'DMZ', 'CORP-LAN').",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Validators
    # ──────────────────────────────────────────────────────────────────────────

    @field_validator("src_ip", "dst_ip", mode="before")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        import ipaddress
        try:
            ipaddress.ip_address(v)
        except ValueError as exc:
            raise ValueError(f"Invalid IP address: {v!r}") from exc
        return v

    @model_validator(mode="after")
    def validate_packet_length_stats(self) -> "FlowFeatureVector":
        if self.min_packet_length > self.max_packet_length:
            raise ValueError(
                f"min_packet_length ({self.min_packet_length}) must be ≤ "
                f"max_packet_length ({self.max_packet_length})."
            )
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Serialisation Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def to_redis_payload(self) -> str:
        """Serialise to JSON string for Redis LPUSH / RPUSH."""
        return self.model_dump_json()

    @classmethod
    def from_redis_payload(cls, raw: str | bytes) -> "FlowFeatureVector":
        """Deserialise from Redis payload (str or bytes)."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls.model_validate_json(raw)

    def to_feature_array(self, feature_columns: List[str]) -> list:
        """
        Extract ordered feature values for ML inference.

        Parameters
        ----------
        feature_columns : list[str]
            Ordered list of field names matching the model's expected input.

        Returns
        -------
        list
            Feature values in the exact order of ``feature_columns``.
        """
        data = self.model_dump()
        return [data[col] for col in feature_columns]

    def to_db_record(self) -> dict:
        """
        Flatten to a dict for TimescaleDB batch insertion.
        Converts datetime → ISO string for async driver compatibility.
        """
        record = self.model_dump()
        record["capture_timestamp"] = self.capture_timestamp.isoformat()
        return record


# ──────────────────────────────────────────────────────────────────────────────
# Inference Result Contract
# ──────────────────────────────────────────────────────────────────────────────

class FlowPrediction(BaseModel):
    """
    Result payload appended by the ML Inference Service.
    Stored in TimescaleDB and broadcast via WebSocket.
    """

    flow_id: str
    predicted_class: AttackClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Softmax/probability scores per class (keys = AttackClass values).",
    )
    model_name: str = Field(..., description="Model identifier, e.g. 'lightgbm-v1.2'.")
    inference_latency_ms: float = Field(..., ge=0.0)
    shap_queued: bool = Field(
        default=False,
        description="True if SHAP explanation has been enqueued for async processing.",
    )
    inference_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ──────────────────────────────────────────────────────────────────────────────
# SHAP Explanation Contract
# ──────────────────────────────────────────────────────────────────────────────

class ShapExplanation(BaseModel):
    """
    SHAP feature-attribution payload produced by the async background worker.
    Linked to ``FlowPrediction`` via ``flow_id``.
    """

    flow_id: str
    shap_values: Dict[str, float] = Field(
        ...,
        description="Feature name → SHAP value mapping (top-k or all features).",
    )
    base_value: float = Field(..., description="SHAP expected-value (model base rate).")
    top_features: List[str] = Field(
        default_factory=list,
        description="Ordered list of top-contributing feature names (descending |SHAP|).",
    )
    explainer_type: str = Field(
        default="TreeExplainer",
        description="SHAP explainer class used (TreeExplainer / DeepExplainer / GradientExplainer).",
    )
    explanation_latency_ms: float = Field(..., ge=0.0)
    explanation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
