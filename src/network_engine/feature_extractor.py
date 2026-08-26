"""
feature_extractor.py
────────────────────
CICFlowMeter-compatible 78-feature extractor for completed network flows.

Design
──────
  FlowRecord              — mutable per-packet accumulator (updated live)
  RawPacketInfo           — lightweight single-packet snapshot
  FlowFeatureExtractor    — pure stateless computation engine
      .extract(record)    → FlowFeatureVector  (Pydantic V2)

All feature definitions follow the original CICFlowMeter Java implementation
semantics so that live-capture output is directly compatible with ML models
trained on CICIDS2017 CICFlowMeter CSV files.

Feature groups
──────────────
  1.  Packet counts & total lengths       (fwd / bwd)
  2.  Packet length statistics            (min / max / mean / std per direction)
  3.  Flow byte & packet rates            (bytes/s, pkts/s, directional)
  4.  Inter-Arrival Times (IAT)           (flow-level + fwd + bwd)
  5.  TCP flag counts                     (aggregate + per-direction PSH/URG)
  6.  Header lengths                      (cumulative fwd / bwd)
  7.  Active / Idle periods               (burst detection, μs)
  8.  Sub-flow metrics                    (simplified: 1 subflow = entire flow)
  9.  TCP window & segment metrics        (init_win, act_data_pkt, min_seg_size)
  10. Bulk transfer metrics               (avg bytes/pkts/rate per bulk burst)
  11. Derived ratios                      (down/up, avg sizes)

Author : Person 3 — Network Engine
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ── Import FlowFeatureVector from the data layer ──────────────────────────────
# Supports both 'python -m src.network_engine.feature_extractor' and
# 'python src/network_engine/feature_extractor.py' invocation styles.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "data"))
try:
    from flow_schema import FlowFeatureVector  # type: ignore
except ImportError:
    from src.data.flow_schema import FlowFeatureVector  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# TCP Flag Bitmasks  (RFC 793 + RFC 3168)
# ─────────────────────────────────────────────────────────────────────────────

class TCPFlags:
    FIN = 0x01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10
    URG = 0x20
    ECE = 0x40
    CWR = 0x80


# ─────────────────────────────────────────────────────────────────────────────
# RawPacketInfo — Lightweight single-packet snapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawPacketInfo:
    """
    Minimal per-packet data extracted by the packet parser.
    Stored verbatim inside FlowRecord for later feature computation.
    """

    timestamp:    float
    """Epoch seconds (float; Scapy provides microsecond precision)."""

    length:       int
    """Total IP-layer length in bytes (IP header + transport header + payload)."""

    header_len:   int
    """Combined IP + transport header size in bytes."""

    is_forward:   bool
    """
    True when this packet travels in the same direction as the first packet
    of the flow (i.e., from flow initiator to responder).
    Set to True initially by the parser; corrected by FlowAggregator.
    """

    # ── TCP-specific fields (0 / -1 for non-TCP) ─────────────────────────────
    tcp_flags:    int = 0
    """Raw TCP flags byte (OR of TCPFlags constants)."""

    tcp_window:   int = 0
    """TCP receive window size (16-bit, from the TCP header)."""

    tcp_data_len: int = 0
    """TCP payload length = length − header_len (0 for pure-ACK / non-TCP)."""


# ─────────────────────────────────────────────────────────────────────────────
# FlowRecord — Mutable per-flow accumulator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowRecord:
    """
    Stateful accumulator for an active network flow.

    One FlowRecord lives in FlowAggregator's flow table for the duration
    of the flow.  When the flow expires (timeout / TCP FIN / TCP RST),
    FlowFeatureExtractor.extract() converts it to an immutable
    FlowFeatureVector for downstream ML inference.

    Directionality convention
    ─────────────────────────
    ``src_ip`` / ``src_port`` always refer to the *initiating* endpoint
    (first packet's source).  Packets are marked is_forward=True when they
    originate from that endpoint; is_forward=False for the reverse direction.
    """

    # ── 5-tuple identity ──────────────────────────────────────────────────────
    src_ip:   str
    """Initiating endpoint's IP address."""

    dst_ip:   str
    """Responding endpoint's IP address."""

    src_port: int
    """Initiating endpoint's transport port (0 for ICMP)."""

    dst_port: int
    """Responding endpoint's transport port (0 for ICMP)."""

    protocol: int
    """IP protocol number: 6 = TCP, 17 = UDP, 1 = ICMP."""

    # ── Timing ────────────────────────────────────────────────────────────────
    start_time: float = 0.0
    """Epoch timestamp of the first packet in this flow (seconds)."""

    last_time:  float = 0.0
    """Epoch timestamp of the most recently seen packet (seconds)."""

    # ── Packet list ───────────────────────────────────────────────────────────
    packets: List[RawPacketInfo] = field(default_factory=list)
    """Ordered list of all packets in this flow (appended per-packet)."""

    # ── TCP handshake tracking ────────────────────────────────────────────────
    init_win_fwd: int = -1
    """TCP window size from the first SYN in the forward direction (-1 = not seen)."""

    init_win_bwd: int = -1
    """TCP window size from the first SYN-ACK in the backward direction."""

    fwd_syn_seen: bool = False
    bwd_syn_seen: bool = False

    # ── Optional capture metadata ─────────────────────────────────────────────
    sensor_id:       Optional[str] = None
    network_segment: Optional[str] = None

    # ── Derived views (computed properties, not stored) ───────────────────────

    @property
    def fwd_packets(self) -> List[RawPacketInfo]:
        """All forward-direction packets."""
        return [p for p in self.packets if p.is_forward]

    @property
    def bwd_packets(self) -> List[RawPacketInfo]:
        """All backward-direction packets."""
        return [p for p in self.packets if not p.is_forward]

    @property
    def duration_seconds(self) -> float:
        """Flow duration in seconds (last − first packet timestamp)."""
        return max(0.0, self.last_time - self.start_time)

    @property
    def duration_us(self) -> float:
        """Flow duration in microseconds (CICFlowMeter unit)."""
        return self.duration_seconds * 1_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal computation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Threshold that separates an "active" burst from an "idle" gap.
# Matches CICFlowMeter's hardcoded 1-second boundary.
_IDLE_THRESHOLD_S: float = 1.0


def _safe_stats(values: List[float]) -> Tuple[float, float, float, float]:
    """
    Return (mean, std, max, min) for a list of floats.
    Returns (0.0, 0.0, 0.0, 0.0) for empty or single-element lists to avoid
    division-by-zero / undefined variance.
    """
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return (
        float(arr.mean()),
        float(arr.std(ddof=0)),   # population std — matches CICFlowMeter
        float(arr.max()),
        float(arr.min()),
    )


def _iat_deltas_us(timestamps: List[float]) -> List[float]:
    """
    Compute inter-arrival time (IAT) deltas in microseconds from a list
    of epoch-second timestamps.

    Parameters
    ----------
    timestamps : list[float]
        Unsorted list of packet arrival times (epoch seconds).

    Returns
    -------
    list[float]
        Consecutive deltas in microseconds.  Empty if fewer than 2 timestamps.
    """
    if len(timestamps) < 2:
        return []
    ts = sorted(timestamps)
    return [(ts[i + 1] - ts[i]) * 1_000_000.0 for i in range(len(ts) - 1)]


def _active_idle_periods(
    all_timestamps: List[float],
    idle_threshold: float = _IDLE_THRESHOLD_S,
) -> Tuple[List[float], List[float]]:
    """
    Separate a flow's timeline into "active" bursts and "idle" gaps.

    Algorithm
    ---------
    Walk sorted timestamps.  When the gap between two consecutive packets
    exceeds ``idle_threshold`` seconds, the current burst ends (active period
    recorded) and an idle period is recorded before the next burst begins.

    Parameters
    ----------
    all_timestamps : list[float]
        All packet arrival timestamps in the flow (epoch seconds).
    idle_threshold : float
        Gap duration (seconds) that marks a transition from active to idle.

    Returns
    -------
    active_durations : list[float]  (microseconds)
    idle_durations   : list[float]  (microseconds)
    """
    if len(all_timestamps) < 2:
        return [], []

    ts = sorted(all_timestamps)
    active_durations: List[float] = []
    idle_durations:   List[float] = []

    burst_start = ts[0]
    burst_last  = ts[0]

    for i in range(1, len(ts)):
        gap_s = ts[i] - ts[i - 1]
        if gap_s >= idle_threshold:
            # Close current active burst
            active_dur_us = (burst_last - burst_start) * 1_000_000.0
            if active_dur_us > 0.0:
                active_durations.append(active_dur_us)
            idle_durations.append(gap_s * 1_000_000.0)
            burst_start = ts[i]
        burst_last = ts[i]

    # Close the final burst
    active_dur_us = (burst_last - burst_start) * 1_000_000.0
    if active_dur_us > 0.0:
        active_durations.append(active_dur_us)

    return active_durations, idle_durations


def _bulk_metrics(
    packets: List[RawPacketInfo],
) -> Tuple[float, float, float]:
    """
    Compute bulk transfer metrics for one direction's packet list.

    A "bulk" is a sequence of >= 4 consecutive packets that each carry
    TCP payload (tcp_data_len > 0).  This matches CICFlowMeter's minimum
    bulk threshold.

    Parameters
    ----------
    packets : list[RawPacketInfo]
        All packets from a single direction (fwd or bwd).

    Returns
    -------
    avg_bytes_per_bulk   : float  — average payload bytes per bulk burst
    avg_packets_per_bulk : float  — average packet count per bulk burst
    avg_bulk_rate        : float  — average bytes/second across bursts
    """
    BULK_TRIGGER = 4   # minimum payload packets to constitute a bulk

    bulk_count          = 0
    bulk_total_bytes    = 0
    bulk_total_pkts     = 0
    bulk_total_duration = 0.0   # seconds

    run_pkts  = 0
    run_bytes = 0
    run_start = 0.0
    run_last  = 0.0

    def _commit_run() -> None:
        nonlocal bulk_count, bulk_total_bytes, bulk_total_pkts, bulk_total_duration
        if run_pkts >= BULK_TRIGGER:
            bulk_count          += 1
            bulk_total_bytes    += run_bytes
            bulk_total_pkts     += run_pkts
            bulk_total_duration += max(run_last - run_start, 1e-9)

    for pkt in packets:
        if pkt.tcp_data_len > 0:
            if run_pkts == 0:
                run_start = pkt.timestamp
            run_pkts  += 1
            run_bytes += pkt.tcp_data_len
            run_last   = pkt.timestamp
        else:
            _commit_run()
            run_pkts  = 0
            run_bytes = 0

    _commit_run()   # flush the last run

    if bulk_count == 0:
        return 0.0, 0.0, 0.0

    return (
        bulk_total_bytes   / bulk_count,                     # avg bytes/bulk
        bulk_total_pkts    / bulk_count,                     # avg pkts/bulk
        bulk_total_bytes   / bulk_total_duration,            # avg bytes/s
    )


# ─────────────────────────────────────────────────────────────────────────────
# FlowFeatureExtractor — pure stateless computation engine
# ─────────────────────────────────────────────────────────────────────────────

class FlowFeatureExtractor:
    """
    Converts a completed FlowRecord into a FlowFeatureVector.

    Design
    ------
    - Entirely stateless — safe for concurrent use across threads.
    - Raises ValueError if the record has zero packets.
    - All 78 CICIDS2017 features are computed in a single pass where possible.

    Usage
    -----
    extractor = FlowFeatureExtractor()
    vector = extractor.extract(record)
    redis_payload = vector.to_redis_payload()
    """

    def extract(self, record: FlowRecord) -> FlowFeatureVector:
        """
        Compute all features from a completed FlowRecord.

        Parameters
        ----------
        record : FlowRecord
            Fully-accumulated flow record (at least one packet required).

        Returns
        -------
        FlowFeatureVector
            Pydantic-validated 78-feature vector ready for ML inference.

        Raises
        ------
        ValueError
            If the flow record contains zero packets.
        """
        if not record.packets:
            raise ValueError(
                f"FlowRecord {record.src_ip}:{record.src_port} → "
                f"{record.dst_ip}:{record.dst_port} has no packets."
            )

        fwd = record.fwd_packets
        bwd = record.bwd_packets
        all_pkts = record.packets

        dur_s  = record.duration_seconds
        dur_us = record.duration_us

        # ── 1. Packet length arrays ───────────────────────────────────────────
        fwd_lens = [float(p.length) for p in fwd]
        bwd_lens = [float(p.length) for p in bwd]
        all_lens = [float(p.length) for p in all_pkts]

        # ── 2. Packet counts & total lengths ──────────────────────────────────
        n_fwd = len(fwd)
        n_bwd = len(bwd)
        n_all = len(all_pkts)

        total_len_fwd = float(sum(fwd_lens))
        total_len_bwd = float(sum(bwd_lens))
        total_bytes   = total_len_fwd + total_len_bwd

        # ── 3. Packet length statistics (per direction + aggregate) ───────────
        fwd_mean, fwd_std, fwd_max, fwd_min = _safe_stats(fwd_lens)
        bwd_mean, bwd_std, bwd_max, bwd_min = _safe_stats(bwd_lens)
        all_mean, all_std, all_max, all_min = _safe_stats(all_lens)

        # ── 4. Flow rates ─────────────────────────────────────────────────────
        if dur_s > 0.0:
            flow_bytes_per_sec  = total_bytes / dur_s
            flow_pkts_per_sec   = n_all       / dur_s
            fwd_pkts_per_sec    = n_fwd       / dur_s
            bwd_pkts_per_sec    = n_bwd       / dur_s
        else:
            # Single-packet flows: rates are undefined; use 0
            flow_bytes_per_sec  = 0.0
            flow_pkts_per_sec   = 0.0
            fwd_pkts_per_sec    = 0.0
            bwd_pkts_per_sec    = 0.0

        # ── 5. IAT — flow level ───────────────────────────────────────────────
        all_ts  = [p.timestamp for p in all_pkts]
        all_iat = _iat_deltas_us(all_ts)
        flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _safe_stats(all_iat)

        # ── 6. IAT — forward ──────────────────────────────────────────────────
        fwd_ts   = [p.timestamp for p in fwd]
        fwd_iats = _iat_deltas_us(fwd_ts)
        fwd_iat_total = float(sum(fwd_iats))
        fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _safe_stats(fwd_iats)

        # ── 7. IAT — backward ─────────────────────────────────────────────────
        bwd_ts   = [p.timestamp for p in bwd]
        bwd_iats = _iat_deltas_us(bwd_ts)
        bwd_iat_total = float(sum(bwd_iats))
        bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _safe_stats(bwd_iats)

        # ── 8. TCP flags ──────────────────────────────────────────────────────
        fwd_psh = sum(1 for p in fwd      if p.tcp_flags & TCPFlags.PSH)
        bwd_psh = sum(1 for p in bwd      if p.tcp_flags & TCPFlags.PSH)
        fwd_urg = sum(1 for p in fwd      if p.tcp_flags & TCPFlags.URG)
        bwd_urg = sum(1 for p in bwd      if p.tcp_flags & TCPFlags.URG)

        fin_cnt = sum(1 for p in all_pkts if p.tcp_flags & TCPFlags.FIN)
        syn_cnt = sum(1 for p in all_pkts if p.tcp_flags & TCPFlags.SYN)
        rst_cnt = sum(1 for p in all_pkts if p.tcp_flags & TCPFlags.RST)
        psh_cnt = sum(1 for p in all_pkts if p.tcp_flags & TCPFlags.PSH)
        ack_cnt = sum(1 for p in all_pkts if p.tcp_flags & TCPFlags.ACK)
        urg_cnt = sum(1 for p in all_pkts if p.tcp_flags & TCPFlags.URG)
        ece_cnt = sum(1 for p in all_pkts if p.tcp_flags & TCPFlags.ECE)
        cwr_cnt = sum(1 for p in all_pkts if p.tcp_flags & TCPFlags.CWR)

        # ── 9. Header lengths (cumulative) ────────────────────────────────────
        fwd_hdr_total = sum(p.header_len for p in fwd)
        bwd_hdr_total = sum(p.header_len for p in bwd)

        # ── 10. Active / Idle periods ─────────────────────────────────────────
        active_durs, idle_durs = _active_idle_periods(all_ts)
        act_mean, act_std, act_max, act_min   = _safe_stats(active_durs)
        idle_mean, idle_std, idle_max, idle_min = _safe_stats(idle_durs)

        # ── 11. Sub-flow metrics (simplified: 1 subflow = entire flow) ────────
        # CICFlowMeter divides totals by the number of subflows; with a single
        # active period the subflow equals the whole flow.
        subflow_fwd_pkts  = n_fwd
        subflow_fwd_bytes = total_len_fwd
        subflow_bwd_pkts  = n_bwd
        subflow_bwd_bytes = total_len_bwd

        # ── 12. TCP window / segment metrics ──────────────────────────────────
        # act_data_pkt_fwd: forward packets that carried actual TCP payload
        act_data_pkt_fwd = sum(1 for p in fwd if p.tcp_data_len > 0)

        # min_seg_size_fwd: smallest non-zero TCP payload in forward direction
        fwd_data_lens = [p.tcp_data_len for p in fwd if p.tcp_data_len > 0]
        min_seg_size_fwd = int(min(fwd_data_lens)) if fwd_data_lens else 0

        # ── 13. Bulk transfer metrics ─────────────────────────────────────────
        fwd_avg_bpb, fwd_avg_ppb, fwd_avg_br = _bulk_metrics(fwd)
        bwd_avg_bpb, bwd_avg_ppb, bwd_avg_br = _bulk_metrics(bwd)

        # ── 14. Derived ratios ────────────────────────────────────────────────
        down_up_ratio = float(n_bwd) / float(n_fwd) if n_fwd > 0 else 0.0
        avg_pkt_size  = (total_bytes  / n_all) if n_all > 0 else 0.0
        avg_fwd_seg   = (total_len_fwd / n_fwd) if n_fwd > 0 else 0.0
        avg_bwd_seg   = (total_len_bwd / n_bwd) if n_bwd > 0 else 0.0

        # ── Assemble and return the validated FlowFeatureVector ───────────────
        return FlowFeatureVector(
            # Identity
            src_ip   = record.src_ip,
            dst_ip   = record.dst_ip,
            src_port = record.src_port,
            dst_port = record.dst_port,
            protocol = record.protocol,

            # Duration
            flow_duration = dur_us,

            # Packet counts
            total_fwd_packets        = n_fwd,
            total_bwd_packets        = n_bwd,
            total_length_fwd_packets = total_len_fwd,
            total_length_bwd_packets = total_len_bwd,

            # Fwd packet length stats
            fwd_packet_length_max  = fwd_max,
            fwd_packet_length_min  = fwd_min,
            fwd_packet_length_mean = fwd_mean,
            fwd_packet_length_std  = fwd_std,

            # Bwd packet length stats
            bwd_packet_length_max  = bwd_max,
            bwd_packet_length_min  = bwd_min,
            bwd_packet_length_mean = bwd_mean,
            bwd_packet_length_std  = bwd_std,

            # Flow rates
            flow_bytes_per_sec   = flow_bytes_per_sec,
            flow_packets_per_sec = flow_pkts_per_sec,

            # IAT — flow level
            flow_iat_mean = flow_iat_mean,
            flow_iat_std  = flow_iat_std,
            flow_iat_max  = flow_iat_max,
            flow_iat_min  = flow_iat_min,

            # IAT — forward
            fwd_iat_total = fwd_iat_total,
            fwd_iat_mean  = fwd_iat_mean,
            fwd_iat_std   = fwd_iat_std,
            fwd_iat_max   = fwd_iat_max,
            fwd_iat_min   = fwd_iat_min,

            # IAT — backward
            bwd_iat_total = bwd_iat_total,
            bwd_iat_mean  = bwd_iat_mean,
            bwd_iat_std   = bwd_iat_std,
            bwd_iat_max   = bwd_iat_max,
            bwd_iat_min   = bwd_iat_min,

            # TCP flags
            fwd_psh_flags  = fwd_psh,
            bwd_psh_flags  = bwd_psh,
            fwd_urg_flags  = fwd_urg,
            bwd_urg_flags  = bwd_urg,
            fin_flag_count = fin_cnt,
            syn_flag_count = syn_cnt,
            rst_flag_count = rst_cnt,
            psh_flag_count = psh_cnt,
            ack_flag_count = ack_cnt,
            urg_flag_count = urg_cnt,
            cwe_flag_count = cwr_cnt,
            ece_flag_count = ece_cnt,

            # Header lengths
            fwd_header_length = fwd_hdr_total,
            bwd_header_length = bwd_hdr_total,

            # Directional rates
            fwd_packets_per_sec = fwd_pkts_per_sec,
            bwd_packets_per_sec = bwd_pkts_per_sec,

            # Aggregate packet length stats
            min_packet_length      = all_min,
            max_packet_length      = all_max,
            packet_length_mean     = all_mean,
            packet_length_std      = all_std,
            packet_length_variance = all_std ** 2,

            # Active / Idle periods
            active_mean = act_mean,
            active_std  = act_std,
            active_max  = act_max,
            active_min  = act_min,
            idle_mean   = idle_mean,
            idle_std    = idle_std,
            idle_max    = idle_max,
            idle_min    = idle_min,

            # Sub-flow metrics
            subflow_fwd_packets = subflow_fwd_pkts,
            subflow_fwd_bytes   = subflow_fwd_bytes,
            subflow_bwd_packets = subflow_bwd_pkts,
            subflow_bwd_bytes   = subflow_bwd_bytes,

            # TCP window / segment metrics
            init_win_bytes_fwd = record.init_win_fwd,
            init_win_bytes_bwd = record.init_win_bwd,
            act_data_pkt_fwd   = act_data_pkt_fwd,
            min_seg_size_fwd   = min_seg_size_fwd,

            # Bulk metrics
            fwd_avg_bytes_per_bulk    = fwd_avg_bpb,
            fwd_avg_packets_per_bulk  = fwd_avg_ppb,
            fwd_avg_bulk_rate         = fwd_avg_br,
            bwd_avg_bytes_per_bulk    = bwd_avg_bpb,
            bwd_avg_packets_per_bulk  = bwd_avg_ppb,
            bwd_avg_bulk_rate         = bwd_avg_br,

            # Derived ratios
            down_up_ratio        = down_up_ratio,
            avg_packet_size      = avg_pkt_size,
            avg_fwd_segment_size = avg_fwd_seg,
            avg_bwd_segment_size = avg_bwd_seg,

            # Capture metadata
            sensor_id       = record.sensor_id,
            network_segment = record.network_segment,
        )
