"""
test_feature_extractor.py
─────────────────────────
Unit tests for the Network Engine feature extractor.

Tests the full feature computation pipeline using hand-crafted FlowRecord
objects — no Scapy or Redis required.

Coverage
────────
  - Single-packet flow (edge-case: zero IAT, zero duration)
  - Forward-only flow (multi-packet)
  - Bidirectional flow (fwd + bwd packets)
  - TCP flag extraction (PSH, URG, SYN, FIN, ACK, RST, CWR, ECE)
  - IAT delta computation
  - Active / idle period detection
  - Bulk transfer metric detection
  - Feature vector Pydantic validation
  - Redis serialisation round-trip of the output FlowFeatureVector
  - Zero-duration guard (no divide-by-zero)
  - Empty packet list raises ValueError

Run
───
    pytest tests/unit/test_feature_extractor.py -v
"""

from __future__ import annotations

import math
import sys
import time

import pytest

sys.path.insert(0, "src/network_engine")
sys.path.insert(0, "src/data")

from feature_extractor import (
    FlowFeatureExtractor,
    FlowRecord,
    RawPacketInfo,
    TCPFlags,
    _iat_deltas_us,
    _active_idle_periods,
    _bulk_metrics,
    _safe_stats,
)
from flow_schema import FlowFeatureVector


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTOR = FlowFeatureExtractor()
_T0 = 1_700_000_000.0   # fixed epoch anchor for deterministic tests


def _pkt(
    offset_s:     float,
    length:       int,
    header_len:   int  = 40,
    is_forward:   bool = True,
    tcp_flags:    int  = 0,
    tcp_window:   int  = 0,
    tcp_data_len: int  = 0,
) -> RawPacketInfo:
    """Convenience factory for RawPacketInfo."""
    return RawPacketInfo(
        timestamp    = _T0 + offset_s,
        length       = length,
        header_len   = header_len,
        is_forward   = is_forward,
        tcp_flags    = tcp_flags,
        tcp_window   = tcp_window,
        tcp_data_len = tcp_data_len,
    )


def _tcp_pkt(
    offset_s:  float,
    length:    int,
    flags:     int,
    is_forward: bool = True,
    window:    int  = 65535,
    data_len:  int  = 0,
) -> RawPacketInfo:
    return _pkt(
        offset_s     = offset_s,
        length       = length,
        header_len   = 40,
        is_forward   = is_forward,
        tcp_flags    = flags,
        tcp_window   = window,
        tcp_data_len = data_len,
    )


def _make_record(packets, proto=6) -> FlowRecord:
    """Build a minimal FlowRecord from a packet list."""
    rec = FlowRecord(
        src_ip   = "10.0.0.1",
        dst_ip   = "10.0.0.2",
        src_port = 12345,
        dst_port = 443,
        protocol = proto,
    )
    rec.packets    = packets
    rec.start_time = packets[0].timestamp
    rec.last_time  = packets[-1].timestamp
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# Helper-function unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeStats:
    def test_empty_list_returns_zeros(self):
        assert _safe_stats([]) == (0.0, 0.0, 0.0, 0.0)

    def test_single_value(self):
        mean, std, mx, mn = _safe_stats([5.0])
        assert mean == 5.0
        assert std  == 0.0
        assert mx   == 5.0
        assert mn   == 5.0

    def test_multiple_values(self):
        values = [2.0, 4.0, 6.0]
        mean, std, mx, mn = _safe_stats(values)
        assert mean == pytest.approx(4.0)
        assert mx   == 6.0
        assert mn   == 2.0


class TestIATDeltas:
    def test_empty(self):
        assert _iat_deltas_us([]) == []

    def test_single(self):
        assert _iat_deltas_us([_T0]) == []

    def test_two_packets(self):
        # 0.5 s gap → 500_000 µs
        deltas = _iat_deltas_us([_T0, _T0 + 0.5])
        assert len(deltas) == 1
        assert deltas[0] == pytest.approx(500_000.0)

    def test_out_of_order_input_is_sorted(self):
        deltas = _iat_deltas_us([_T0 + 1.0, _T0, _T0 + 2.0])
        assert len(deltas) == 2
        assert deltas[0] == pytest.approx(1_000_000.0)
        assert deltas[1] == pytest.approx(1_000_000.0)


class TestActiveIdlePeriods:
    def test_empty(self):
        active, idle = _active_idle_periods([])
        assert active == []
        assert idle   == []

    def test_single_packet_no_periods(self):
        active, idle = _active_idle_periods([_T0])
        assert active == []
        assert idle   == []

    def test_continuous_burst_no_idle(self):
        # 3 packets 0.1 s apart — all within 1 s threshold, no idle period
        ts = [_T0, _T0 + 0.1, _T0 + 0.2]
        active, idle = _active_idle_periods(ts)
        assert len(idle) == 0
        assert len(active) == 1
        assert active[0] == pytest.approx(0.2 * 1e6)   # 200_000 µs

    def test_two_bursts_one_idle(self):
        # Burst A: t=0, 0.1 — then 2s idle — Burst B: t=2.1, 2.2
        ts = [_T0, _T0 + 0.1, _T0 + 2.1, _T0 + 2.2]
        active, idle = _active_idle_periods(ts)
        assert len(idle)   == 1
        assert len(active) == 2
        assert idle[0] == pytest.approx(2.0 * 1e6, rel=1e-3)


class TestBulkMetrics:
    def test_no_payload_returns_zeros(self):
        pkts = [_pkt(i * 0.01, 60) for i in range(10)]  # tcp_data_len=0
        assert _bulk_metrics(pkts) == (0.0, 0.0, 0.0)

    def test_fewer_than_trigger_returns_zeros(self):
        pkts = [
            _tcp_pkt(i * 0.01, 100, TCPFlags.ACK, data_len=60)
            for i in range(3)   # only 3 — below BULK_TRIGGER of 4
        ]
        assert _bulk_metrics(pkts) == (0.0, 0.0, 0.0)

    def test_single_bulk_detected(self):
        pkts = [
            _tcp_pkt(i * 0.01, 1500, TCPFlags.ACK, data_len=1460)
            for i in range(6)
        ]
        bpb, ppb, rate = _bulk_metrics(pkts)
        assert bpb  == pytest.approx(6 * 1460, rel=1e-3)
        assert ppb  == pytest.approx(6.0,      rel=1e-3)
        assert rate  > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FlowFeatureExtractor integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractorEdgeCases:
    def test_empty_record_raises(self):
        rec = FlowRecord("1.1.1.1", "2.2.2.2", 80, 443, 6)
        with pytest.raises(ValueError, match="no packets"):
            _EXTRACTOR.extract(rec)

    def test_single_packet_no_divide_by_zero(self):
        pkts = [_pkt(0.0, 100)]
        rec  = _make_record(pkts)
        vec  = _EXTRACTOR.extract(rec)

        assert vec.flow_duration       == pytest.approx(0.0)
        assert vec.flow_bytes_per_sec  == pytest.approx(0.0)
        assert vec.flow_packets_per_sec == pytest.approx(0.0)
        assert vec.total_fwd_packets   == 1
        assert vec.total_bwd_packets   == 0


class TestForwardOnlyFlow:
    """Multi-packet flow with only forward-direction packets."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        pkts = [
            _pkt(0.000, 100),
            _pkt(0.100, 120),
            _pkt(0.250, 80),
            _pkt(0.400, 200),
        ]
        rec = _make_record(pkts)
        self.vec = _EXTRACTOR.extract(rec)
        self.pkts = pkts

    def test_packet_counts(self):
        assert self.vec.total_fwd_packets == 4
        assert self.vec.total_bwd_packets == 0

    def test_total_lengths(self):
        assert self.vec.total_length_fwd_packets == pytest.approx(500.0)
        assert self.vec.total_length_bwd_packets == pytest.approx(0.0)

    def test_flow_duration_us(self):
        # 0.4 s = 400_000 µs
        assert self.vec.flow_duration == pytest.approx(400_000.0, rel=1e-4)

    def test_fwd_length_stats(self):
        assert self.vec.fwd_packet_length_max  == pytest.approx(200.0)
        assert self.vec.fwd_packet_length_min  == pytest.approx(80.0)
        assert self.vec.fwd_packet_length_mean == pytest.approx(125.0)

    def test_bwd_stats_are_zero(self):
        assert self.vec.bwd_packet_length_max  == 0.0
        assert self.vec.bwd_packet_length_min  == 0.0
        assert self.vec.bwd_packet_length_mean == 0.0
        assert self.vec.bwd_packet_length_std  == 0.0

    def test_flow_iat_count(self):
        # 4 packets → 3 IAT deltas
        # We verify stats are reasonable
        assert self.vec.flow_iat_mean > 0.0
        assert self.vec.flow_iat_max  >= self.vec.flow_iat_min

    def test_fwd_iat_total(self):
        # 0→0.1→0.25→0.4: deltas = 0.1, 0.15, 0.15 s → total ≈ 400_000 µs
        assert self.vec.fwd_iat_total == pytest.approx(400_000.0, rel=1e-4)

    def test_bwd_iat_all_zero(self):
        assert self.vec.bwd_iat_total == 0.0
        assert self.vec.bwd_iat_mean  == 0.0

    def test_down_up_ratio_zero(self):
        # 0 bwd / 4 fwd
        assert self.vec.down_up_ratio == pytest.approx(0.0)

    def test_avg_packet_size(self):
        assert self.vec.avg_packet_size == pytest.approx(500.0 / 4)

    def test_flow_rates_positive(self):
        assert self.vec.flow_bytes_per_sec   > 0.0
        assert self.vec.flow_packets_per_sec > 0.0

    def test_returns_valid_pydantic_model(self):
        assert isinstance(self.vec, FlowFeatureVector)


class TestBidirectionalFlow:
    """Flow with both forward and backward packets."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        # 5 fwd + 3 bwd packets interleaved
        pkts = [
            _pkt(0.000, 100,  is_forward=True),
            _pkt(0.050, 80,   is_forward=False),
            _pkt(0.100, 120,  is_forward=True),
            _pkt(0.150, 90,   is_forward=False),
            _pkt(0.200, 110,  is_forward=True),
            _pkt(0.250, 70,   is_forward=False),
            _pkt(0.300, 130,  is_forward=True),
            _pkt(0.400, 140,  is_forward=True),
        ]
        rec = _make_record(pkts)
        self.vec = _EXTRACTOR.extract(rec)
        self.pkts = pkts

    def test_packet_counts(self):
        assert self.vec.total_fwd_packets == 5
        assert self.vec.total_bwd_packets == 3

    def test_down_up_ratio(self):
        # 3 bwd / 5 fwd
        assert self.vec.down_up_ratio == pytest.approx(3.0 / 5.0)

    def test_bwd_length_stats_are_nonzero(self):
        assert self.vec.bwd_packet_length_max > 0.0
        assert self.vec.bwd_packet_length_min > 0.0

    def test_subflow_matches_totals(self):
        assert self.vec.subflow_fwd_packets == self.vec.total_fwd_packets
        assert self.vec.subflow_bwd_packets == self.vec.total_bwd_packets

    def test_aggregate_length_stats(self):
        all_lens = [100, 80, 120, 90, 110, 70, 130, 140]
        assert self.vec.min_packet_length == pytest.approx(float(min(all_lens)))
        assert self.vec.max_packet_length == pytest.approx(float(max(all_lens)))

    def test_packet_length_variance(self):
        # variance == std²
        assert self.vec.packet_length_variance == pytest.approx(
            self.vec.packet_length_std ** 2, rel=1e-5
        )


class TestTCPFlags:
    """TCP flag counting (aggregate + directional)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        pkts = [
            # SYN (fwd)
            _tcp_pkt(0.00, 60,  TCPFlags.SYN,           is_forward=True,  window=65535),
            # SYN-ACK (bwd)
            _tcp_pkt(0.01, 60,  TCPFlags.SYN | TCPFlags.ACK, is_forward=False, window=29200),
            # ACK (fwd)
            _tcp_pkt(0.02, 60,  TCPFlags.ACK,           is_forward=True,  data_len=0),
            # Data + PSH (fwd)
            _tcp_pkt(0.05, 500, TCPFlags.PSH | TCPFlags.ACK, is_forward=True,  data_len=460),
            # Data + PSH (bwd)
            _tcp_pkt(0.10, 200, TCPFlags.PSH | TCPFlags.ACK, is_forward=False, data_len=160),
            # URG (fwd — rare)
            _tcp_pkt(0.15, 60,  TCPFlags.URG | TCPFlags.ACK, is_forward=True),
            # FIN-ACK (fwd)
            _tcp_pkt(0.20, 60,  TCPFlags.FIN | TCPFlags.ACK, is_forward=True),
        ]
        rec = _make_record(pkts)
        rec.init_win_fwd = 65535
        rec.init_win_bwd = 29200
        self.vec = _EXTRACTOR.extract(rec)

    def test_syn_count(self):
        assert self.vec.syn_flag_count == 2   # SYN + SYN-ACK

    def test_fin_count(self):
        assert self.vec.fin_flag_count == 1

    def test_ack_count(self):
        # SYN-ACK (has ACK), pure ACK, PSH+ACK (fwd), PSH+ACK (bwd), URG+ACK, FIN+ACK = 6
        assert self.vec.ack_flag_count == 6

    def test_psh_count(self):
        assert self.vec.psh_flag_count == 2

    def test_fwd_psh_count(self):
        assert self.vec.fwd_psh_flags == 1

    def test_bwd_psh_count(self):
        assert self.vec.bwd_psh_flags == 1

    def test_fwd_urg_count(self):
        assert self.vec.fwd_urg_flags == 1

    def test_bwd_urg_count(self):
        assert self.vec.bwd_urg_flags == 0

    def test_init_win_bytes_fwd(self):
        assert self.vec.init_win_bytes_fwd == 65535

    def test_init_win_bytes_bwd(self):
        assert self.vec.init_win_bytes_bwd == 29200

    def test_act_data_pkt_fwd(self):
        # Only the PSH+ACK fwd packet has data_len > 0
        assert self.vec.act_data_pkt_fwd == 1

    def test_min_seg_size_fwd(self):
        assert self.vec.min_seg_size_fwd == 460


class TestActiveIdleInVector:
    """Active and idle period computation end-to-end."""

    def test_two_bursts_produce_idle_period(self):
        # Burst A at t=0, 0.1  → idle of 3 s → Burst B at t=3.1, 3.2
        pkts = [
            _pkt(0.0, 100),
            _pkt(0.1, 100),
            _pkt(3.1, 100),
            _pkt(3.2, 100),
        ]
        rec = _make_record(pkts)
        vec = _EXTRACTOR.extract(rec)

        assert vec.idle_mean  > 0.0
        assert vec.idle_max   > 0.0
        assert vec.active_mean > 0.0

    def test_single_burst_no_idle(self):
        pkts = [_pkt(i * 0.05, 100) for i in range(4)]
        rec  = _make_record(pkts)
        vec  = _EXTRACTOR.extract(rec)

        # No gap exceeds 1 s threshold → no idle periods
        assert vec.idle_mean == 0.0
        assert vec.idle_max  == 0.0


class TestRedisRoundTrip:
    """FlowFeatureVector produced by extractor must survive Redis serialisation."""

    def test_redis_roundtrip(self):
        pkts = [
            _pkt(0.0,   200, is_forward=True),
            _pkt(0.05,  150, is_forward=False),
            _pkt(0.10,  180, is_forward=True),
            _pkt(0.20,  100, is_forward=False),
        ]
        rec = _make_record(pkts)
        vec = _EXTRACTOR.extract(rec)

        payload  = vec.to_redis_payload()
        restored = FlowFeatureVector.from_redis_payload(payload)

        assert restored.flow_id          == vec.flow_id
        assert restored.src_ip           == vec.src_ip
        assert restored.total_fwd_packets == vec.total_fwd_packets
        assert restored.flow_duration     == pytest.approx(vec.flow_duration)

    def test_to_db_record_has_iso_timestamp(self):
        pkts = [_pkt(0.0, 100)]
        rec  = _make_record(pkts)
        vec  = _EXTRACTOR.extract(rec)
        db   = vec.to_db_record()

        from datetime import datetime
        # Must not raise
        datetime.fromisoformat(db["capture_timestamp"])
