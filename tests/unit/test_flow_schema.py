"""
test_flow_schema.py
───────────────────
Unit tests for the FlowFeatureVector Pydantic V2 schema.
Validates field constraints, IP validation, and serialisation round-trips.

Run:  pytest tests/unit/test_flow_schema.py -v
"""
import json
import sys
sys.path.insert(0, "src/data")

import pytest
from pydantic import ValidationError

from flow_schema import AttackClass, FlowFeatureVector, FlowPrediction


# ── Minimal valid payload factory ────────────────────────────────────────────

def _minimal_vector(**overrides) -> dict:
    base = dict(
        src_ip="192.168.1.10",
        dst_ip="8.8.8.8",
        src_port=54321,
        dst_port=443,
        protocol=6,
        flow_duration=1_000_000.0,
        total_fwd_packets=10,
        total_bwd_packets=8,
        total_length_fwd_packets=1400.0,
        total_length_bwd_packets=800.0,
        fwd_packet_length_max=140.0,
        fwd_packet_length_min=20.0,
        fwd_packet_length_mean=70.0,
        fwd_packet_length_std=30.0,
        bwd_packet_length_max=100.0,
        bwd_packet_length_min=20.0,
        bwd_packet_length_mean=50.0,
        bwd_packet_length_std=20.0,
        flow_bytes_per_sec=2200.0,
        flow_packets_per_sec=18.0,
        flow_iat_mean=55_000.0,
        flow_iat_std=5_000.0,
        flow_iat_max=200_000.0,
        flow_iat_min=1_000.0,
        fwd_iat_total=900_000.0,
        fwd_iat_mean=100_000.0,
        fwd_iat_std=10_000.0,
        fwd_iat_max=300_000.0,
        fwd_iat_min=5_000.0,
        bwd_iat_total=700_000.0,
        bwd_iat_mean=87_500.0,
        bwd_iat_std=8_000.0,
        bwd_iat_max=250_000.0,
        bwd_iat_min=4_000.0,
        fwd_psh_flags=1, bwd_psh_flags=0,
        fwd_urg_flags=0, bwd_urg_flags=0,
        fin_flag_count=1, syn_flag_count=1,
        rst_flag_count=0, psh_flag_count=1,
        ack_flag_count=12, urg_flag_count=0,
        cwe_flag_count=0, ece_flag_count=0,
        fwd_header_length=200,
        bwd_header_length=160,
        fwd_packets_per_sec=10.0,
        bwd_packets_per_sec=8.0,
        min_packet_length=20.0,
        max_packet_length=140.0,
        packet_length_mean=62.0,
        packet_length_std=25.0,
        packet_length_variance=625.0,
        active_mean=500_000.0, active_std=50_000.0,
        active_max=800_000.0, active_min=200_000.0,
        idle_mean=100_000.0, idle_std=10_000.0,
        idle_max=300_000.0, idle_min=10_000.0,
        subflow_fwd_packets=10, subflow_fwd_bytes=1400.0,
        subflow_bwd_packets=8,  subflow_bwd_bytes=800.0,
        init_win_bytes_fwd=65535, init_win_bytes_bwd=65535,
        act_data_pkt_fwd=9, min_seg_size_fwd=20,
        down_up_ratio=0.8,
        avg_packet_size=62.0,
        avg_fwd_segment_size=70.0,
        avg_bwd_segment_size=50.0,
        fwd_avg_bytes_per_bulk=0.0,
        fwd_avg_packets_per_bulk=0.0,
        fwd_avg_bulk_rate=0.0,
        bwd_avg_bytes_per_bulk=0.0,
        bwd_avg_packets_per_bulk=0.0,
        bwd_avg_bulk_rate=0.0,
    )
    base.update(overrides)
    return base


# ── Construction tests ────────────────────────────────────────────────────────

def test_valid_vector_constructs() -> None:
    v = FlowFeatureVector(**_minimal_vector())
    assert v.src_ip == "192.168.1.10"
    assert v.dst_port == 443

def test_ipv6_address_accepted() -> None:
    v = FlowFeatureVector(**_minimal_vector(src_ip="::1", dst_ip="2001:db8::1"))
    assert v.src_ip == "::1"

def test_invalid_src_ip_raises() -> None:
    with pytest.raises(ValidationError, match="Invalid IP"):
        FlowFeatureVector(**_minimal_vector(src_ip="999.999.999.999"))

def test_negative_port_raises() -> None:
    with pytest.raises(ValidationError):
        FlowFeatureVector(**_minimal_vector(src_port=-1))

def test_port_over_65535_raises() -> None:
    with pytest.raises(ValidationError):
        FlowFeatureVector(**_minimal_vector(dst_port=70000))

def test_min_greater_than_max_packet_length_raises() -> None:
    with pytest.raises(ValidationError, match="min_packet_length"):
        FlowFeatureVector(**_minimal_vector(min_packet_length=200.0, max_packet_length=50.0))

def test_negative_flow_duration_raises() -> None:
    with pytest.raises(ValidationError):
        FlowFeatureVector(**_minimal_vector(flow_duration=-1.0))


# ── Serialisation round-trip ──────────────────────────────────────────────────

def test_redis_roundtrip() -> None:
    original = FlowFeatureVector(**_minimal_vector())
    payload = original.to_redis_payload()
    restored = FlowFeatureVector.from_redis_payload(payload)
    assert restored.flow_id == original.flow_id
    assert restored.src_ip == original.src_ip

def test_redis_roundtrip_bytes() -> None:
    original = FlowFeatureVector(**_minimal_vector())
    payload_bytes = original.to_redis_payload().encode("utf-8")
    restored = FlowFeatureVector.from_redis_payload(payload_bytes)
    assert restored.dst_ip == original.dst_ip

def test_to_feature_array_ordering() -> None:
    v = FlowFeatureVector(**_minimal_vector())
    cols = ["flow_duration", "total_fwd_packets", "total_bwd_packets"]
    arr = v.to_feature_array(cols)
    assert arr == [v.flow_duration, v.total_fwd_packets, v.total_bwd_packets]

def test_to_db_record_has_iso_timestamp() -> None:
    v = FlowFeatureVector(**_minimal_vector())
    record = v.to_db_record()
    # Must be a valid ISO string
    from datetime import datetime
    datetime.fromisoformat(record["capture_timestamp"])


# ── FlowPrediction ────────────────────────────────────────────────────────────

def test_flow_prediction_constructs() -> None:
    v = FlowFeatureVector(**_minimal_vector())
    p = FlowPrediction(
        flow_id=v.flow_id,
        predicted_class=AttackClass.DOS,
        confidence=0.97,
        model_name="lightgbm-v1.0",
        inference_latency_ms=12.5,
    )
    assert p.predicted_class == "DoS"
    assert 0.0 <= p.confidence <= 1.0

def test_confidence_out_of_range_raises() -> None:
    v = FlowFeatureVector(**_minimal_vector())
    with pytest.raises(ValidationError):
        FlowPrediction(
            flow_id=v.flow_id,
            predicted_class=AttackClass.NORMAL,
            confidence=1.5,  # invalid
            model_name="lgbm",
            inference_latency_ms=5.0,
        )
