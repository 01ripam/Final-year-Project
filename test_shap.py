"""
test_shap.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.workers.shap_worker import explain_flow
from src.data.flow_schema import FlowFeatureVector, FlowPrediction, AttackClass
from src.models.registry import registry
from src.models.dummy import DummyClassifier

# 1. Register dummy model
model = DummyClassifier()
registry.register(model.name, model)

# 2. Mock inputs
vector = FlowFeatureVector(
    src_ip="192.168.1.1", dst_ip="8.8.8.8",
    src_port=1234, dst_port=80, protocol=6,
    flow_duration=100.0,
    total_fwd_packets=10, total_bwd_packets=10,
    total_length_fwd_packets=100.0, total_length_bwd_packets=100.0,
    fwd_packet_length_max=10.0, fwd_packet_length_min=10.0,
    fwd_packet_length_mean=10.0, fwd_packet_length_std=0.0,
    bwd_packet_length_max=10.0, bwd_packet_length_min=10.0,
    bwd_packet_length_mean=10.0, bwd_packet_length_std=0.0,
    flow_bytes_per_sec=200.0, flow_packets_per_sec=20.0,
    flow_iat_mean=10.0, flow_iat_std=0.0, flow_iat_max=10.0, flow_iat_min=10.0,
    fwd_iat_total=100.0, fwd_iat_mean=10.0, fwd_iat_std=0.0, fwd_iat_max=10.0, fwd_iat_min=10.0,
    bwd_iat_total=100.0, bwd_iat_mean=10.0, bwd_iat_std=0.0, bwd_iat_max=10.0, bwd_iat_min=10.0,
    fwd_psh_flags=0, bwd_psh_flags=0, fwd_urg_flags=0, bwd_urg_flags=0,
    fin_flag_count=0, syn_flag_count=0, rst_flag_count=0, psh_flag_count=0,
    ack_flag_count=0, urg_flag_count=0, cwe_flag_count=0, ece_flag_count=0,
    fwd_header_length=200, bwd_header_length=200,
    fwd_packets_per_sec=10.0, bwd_packets_per_sec=10.0,
    min_packet_length=10.0, max_packet_length=10.0, packet_length_mean=10.0,
    packet_length_std=0.0, packet_length_variance=0.0,
    active_mean=0.0, active_std=0.0, active_max=0.0, active_min=0.0,
    idle_mean=0.0, idle_std=0.0, idle_max=0.0, idle_min=0.0,
    subflow_fwd_packets=10, subflow_fwd_bytes=100.0,
    subflow_bwd_packets=10, subflow_bwd_bytes=100.0,
    init_win_bytes_fwd=0, init_win_bytes_bwd=0,
    act_data_pkt_fwd=0, min_seg_size_fwd=20,
    down_up_ratio=1.0, avg_packet_size=10.0,
    avg_fwd_segment_size=10.0, avg_bwd_segment_size=10.0,
    fwd_avg_bytes_per_bulk=0.0, fwd_avg_packets_per_bulk=0.0, fwd_avg_bulk_rate=0.0,
    bwd_avg_bytes_per_bulk=0.0, bwd_avg_packets_per_bulk=0.0, bwd_avg_bulk_rate=0.0
)

prediction = FlowPrediction(
    flow_id=vector.flow_id,
    predicted_class=AttackClass("DoS"),
    confidence=0.99,
    class_probabilities={"Normal": 0.01, "DoS": 0.99, "DDoS": 0.0, "Port Scan": 0.0, "Brute Force": 0.0, "Botnet": 0.0, "Web Attack": 0.0},
    model_name="dummy",
    inference_latency_ms=1.5,
    shap_queued=True
)

from unittest.mock import patch

try:
    with patch("src.workers.db_writer.write_shap.delay"):
        res = explain_flow(vector.model_dump_json(), prediction.model_dump_json())
        print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
