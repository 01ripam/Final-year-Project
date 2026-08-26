"""
config.py
─────────
Configuration dataclass for the Network Engine capture pipeline.

All tuneable parameters live here so PacketCapture is easily
unit-testable (just pass a custom CaptureConfig instance).

Author : Person 3 — Network Engine
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CaptureConfig:
    """
    All tuneable parameters for PacketCapture + FlowAggregator.

    Values can be overridden via constructor kwargs or read from
    environment variables by the CLI entry-point in packet_capture.py.
    """

    # ── Network Interface ─────────────────────────────────────────────────────
    interface: Optional[str] = None
    """NIC to capture on (None = Scapy default / all interfaces)."""

    pcap_file: Optional[str] = None
    """
    Path to a .pcap file for offline replay instead of live capture.
    When set, ``interface`` is ignored and packets are read via rdpcap().
    Useful for testing and CI without a live network.
    """

    bpf_filter: str = "ip"
    """
    Berkeley Packet Filter expression applied to the capture socket.
    Default 'ip' captures all IPv4 traffic.
    Example: 'tcp port 80 or tcp port 443' for HTTP/HTTPS only.
    """

    # ── Flow Timeouts ─────────────────────────────────────────────────────────
    active_timeout: float = 120.0
    """
    Forcibly expire a flow after this many seconds regardless of activity.
    Matches CICFlowMeter's default active timeout (120 s).
    """

    idle_timeout: float = 5.0
    """
    Expire a flow that has seen no new packets for this many seconds.
    Matches CICFlowMeter's default idle timeout (5 s).
    """

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    """Redis connection URL passed to redis.from_url()."""

    redis_key: str = "flow:queue"
    """
    Redis list key — completed FlowFeatureVector JSON payloads are
    LPUSH'd here for consumption by the FastAPI / Celery inference workers.
    """

    # ── Sensor Metadata ───────────────────────────────────────────────────────
    sensor_id: Optional[str] = None
    """
    Sensor / interface label attached to every FlowFeatureVector.
    Examples: 'eth0', 'sensor-dmz-01', 'tap-core-switch'.
    """

    network_segment: Optional[str] = None
    """
    Logical network segment label embedded in each flow vector.
    Examples: 'DMZ', 'CORP-LAN', 'OT-NETWORK'.
    """

    # ── Misc ──────────────────────────────────────────────────────────────────
    max_packets: Optional[int] = None
    """
    Stop after capturing this many raw packets.
    None = run indefinitely.
    Used for offline PCAP replay and integration tests.
    """

    sweeper_interval: float = 1.0
    """
    How often (seconds) the background timeout-sweeper thread wakes up
    to evict idle or active-timeout-exceeded flows.
    """
