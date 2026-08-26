"""
packet_capture.py
─────────────────
Live network packet capture, bidirectional flow aggregation, and Redis publishing.

Pipeline
────────
  Scapy sniff() [live] or rdpcap() [offline PCAP]
      └─► PacketCapture._on_packet(pkt)
              └─► FlowAggregator.ingest(pkt)
                      └─► _parse_packet()           — extract 5-tuple + RawPacketInfo
                      └─► FlowRecord update / create
                      └─► FIN/RST → immediate expiry
  FlowAggregator._sweep_loop()  (background thread, every sweeper_interval seconds)
      └─► evict idle-timeout / active-timeout flows
              └─► FlowFeatureExtractor.extract(record)
              └─► redis.lpush(config.redis_key, json_payload)

Directionality
──────────────
The first packet of a flow establishes the forward direction.
The canonical FlowKey is the lexicographically smaller (IP, port) pair so both
directions of a flow resolve to the same table entry.

Author : Person 3 — Network Engine
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

# ── Scapy — suppress noisy runtime warnings on import ────────────────────────
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
logging.getLogger("scapy.loading").setLevel(logging.ERROR)
from scapy.all import IP, TCP, UDP, ICMP, Packet, sniff, rdpcap  # type: ignore

# ── Internal imports ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "data"))
sys.path.insert(0, _HERE)

try:
    from feature_extractor import FlowFeatureExtractor, FlowRecord, RawPacketInfo, TCPFlags
    from config import CaptureConfig
except ImportError:
    from src.network_engine.feature_extractor import (  # type: ignore
        FlowFeatureExtractor, FlowRecord, RawPacketInfo, TCPFlags,
    )
    from src.network_engine.config import CaptureConfig  # type: ignore

import redis as redis_lib  # type: ignore

logger = logging.getLogger("nids.packet_capture")

# ─────────────────────────────────────────────────────────────────────────────
# Flow Key
# ─────────────────────────────────────────────────────────────────────────────

# (src_ip, dst_ip, src_port, dst_port, protocol) — always in canonical order
FlowKey = Tuple[str, str, int, int, int]


def _canonical_key(
    src_ip:   str,
    dst_ip:   str,
    src_port: int,
    dst_port: int,
    protocol: int,
) -> Tuple[FlowKey, bool]:
    """
    Return a deterministic (canonical) flow key that is direction-independent,
    plus a boolean indicating whether (src_ip, src_port) is the canonical "A" side.

    We define the canonical order as the lexicographically smaller (IP, port)
    pair going first.  This ensures:
        forward  packet (A→B) → same key as
        backward packet (B→A)

    Returns
    -------
    key          : FlowKey
    a_is_source  : bool
        True  when src_ip:src_port is the canonical initiator (A-side).
        False when dst_ip:dst_port is canonical A-side
              (this packet travels B→A, i.e. backward).
    """
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a <= b:
        return (src_ip, dst_ip, src_port, dst_port, protocol), True
    else:
        return (dst_ip, src_ip, dst_port, src_port, protocol), False


# ─────────────────────────────────────────────────────────────────────────────
# Packet Parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_packet(
    pkt: Packet,
) -> Optional[Tuple[str, str, int, int, int, RawPacketInfo]]:
    """
    Extract the 5-tuple and a RawPacketInfo snapshot from a Scapy packet.

    Returns None if the packet is not IP or doesn't have a recognised
    transport layer (TCP / UDP / ICMP).

    The ``is_forward`` field of the returned RawPacketInfo is set to True;
    FlowAggregator.ingest() corrects it relative to the FlowRecord's initiator.
    """
    if not pkt.haslayer(IP):
        return None

    ip        = pkt[IP]
    src_ip    = ip.src
    dst_ip    = ip.dst
    protocol  = ip.proto
    ip_hdr_len = ip.ihl * 4   # IHL is in 32-bit words

    # ── TCP ──────────────────────────────────────────────────────────────────
    if pkt.haslayer(TCP):
        tcp             = pkt[TCP]
        src_port        = tcp.sport
        dst_port        = tcp.dport
        tcp_hdr_len     = tcp.dataofs * 4   # Data Offset is in 32-bit words
        header_len      = ip_hdr_len + tcp_hdr_len
        pkt_len         = len(ip)           # IP-layer total (header + payload)
        tcp_data_len    = max(0, pkt_len - header_len)

        raw = RawPacketInfo(
            timestamp    = float(pkt.time),
            length       = pkt_len,
            header_len   = header_len,
            is_forward   = True,            # corrected later
            tcp_flags    = int(tcp.flags),
            tcp_window   = int(tcp.window),
            tcp_data_len = tcp_data_len,
        )

    # ── UDP ──────────────────────────────────────────────────────────────────
    elif pkt.haslayer(UDP):
        udp          = pkt[UDP]
        src_port     = udp.sport
        dst_port     = udp.dport
        header_len   = ip_hdr_len + 8      # UDP header is always 8 bytes
        pkt_len      = len(ip)
        payload_len  = max(0, pkt_len - header_len)

        raw = RawPacketInfo(
            timestamp    = float(pkt.time),
            length       = pkt_len,
            header_len   = header_len,
            is_forward   = True,
            tcp_data_len = payload_len,
        )

    # ── ICMP ─────────────────────────────────────────────────────────────────
    elif pkt.haslayer(ICMP):
        src_port   = 0
        dst_port   = 0
        header_len = ip_hdr_len + 8        # ICMP fixed header = 8 bytes
        pkt_len    = len(ip)

        raw = RawPacketInfo(
            timestamp  = float(pkt.time),
            length     = pkt_len,
            header_len = header_len,
            is_forward = True,
        )

    else:
        return None   # not a recognised transport layer

    return src_ip, dst_ip, src_port, dst_port, protocol, raw


# ─────────────────────────────────────────────────────────────────────────────
# FlowAggregator
# ─────────────────────────────────────────────────────────────────────────────

class FlowAggregator:
    """
    Maintains an in-memory table of active flows.

    Thread safety
    ─────────────
    All accesses to _table are protected by a single RLock.
    The sweeper runs in a daemon thread and acquires the same lock.

    Flow lifecycle
    ──────────────
    1. First packet → new FlowRecord created, inserted into _table.
    2. Subsequent packets → existing record updated.
    3. Expiry triggers:
         a) TCP FIN or RST flag seen    → immediate expiry
         b) Idle timeout exceeded       → swept by background thread
         c) Active timeout exceeded     → swept by background thread or on-packet
    4. Expired record → FlowFeatureExtractor.extract() → on_flow_complete callback.
    """

    def __init__(
        self,
        config:             CaptureConfig,
        on_flow_complete,   # Callable[[FlowRecord], None]
    ) -> None:
        self._config           = config
        self._on_flow_complete = on_flow_complete
        self._table:           Dict[FlowKey, FlowRecord] = {}
        self._lock             = threading.RLock()
        self._running          = False
        self._sweeper:         Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background timeout-sweeper thread."""
        self._running = True
        self._sweeper = threading.Thread(
            target=self._sweep_loop,
            name="nids-flow-sweeper",
            daemon=True,
        )
        self._sweeper.start()
        logger.info(
            "FlowAggregator started — active_timeout=%.0fs  idle_timeout=%.0fs",
            self._config.active_timeout,
            self._config.idle_timeout,
        )

    def stop(self) -> None:
        """Stop sweeper and flush all remaining flows synchronously."""
        self._running = False
        if self._sweeper and self._sweeper.is_alive():
            self._sweeper.join(timeout=5.0)
        self._flush_all()
        logger.info(
            "FlowAggregator stopped. %d flows remaining — flushed.",
            len(self._table),
        )

    # ── Packet ingestion ──────────────────────────────────────────────────────

    def ingest(self, pkt: Packet) -> None:
        """
        Process one Scapy packet.

        1. Parse 5-tuple + RawPacketInfo.
        2. Look up (or create) the corresponding FlowRecord.
        3. Determine direction; update TCP handshake metadata.
        4. Append to flow; update last_time.
        5. Trigger immediate expiry on TCP FIN / RST.
        6. Trigger expiry if active timeout exceeded.
        """
        parsed = _parse_packet(pkt)
        if parsed is None:
            return

        src_ip, dst_ip, src_port, dst_port, protocol, raw = parsed
        key, a_is_source = _canonical_key(src_ip, dst_ip, src_port, dst_port, protocol)
        now = float(pkt.time)

        with self._lock:
            # ── Create new flow ────────────────────────────────────────────
            if key not in self._table:
                # The canonical A-side is the initiator of this new flow.
                # If a_is_source, src_ip IS the initiator; otherwise dst_ip is.
                if a_is_source:
                    rec_src_ip, rec_dst_ip     = src_ip, dst_ip
                    rec_src_port, rec_dst_port = src_port, dst_port
                else:
                    rec_src_ip, rec_dst_ip     = dst_ip, src_ip
                    rec_src_port, rec_dst_port = dst_port, src_port

                record = FlowRecord(
                    src_ip          = rec_src_ip,
                    dst_ip          = rec_dst_ip,
                    src_port        = rec_src_port,
                    dst_port        = rec_dst_port,
                    protocol        = protocol,
                    start_time      = now,
                    last_time       = now,
                    sensor_id       = self._config.sensor_id,
                    network_segment = self._config.network_segment,
                )
                self._table[key] = record

            record = self._table[key]

            # ── Determine direction relative to FlowRecord initiator ───────
            is_forward = (
                src_ip   == record.src_ip   and
                src_port == record.src_port
            )
            raw.is_forward = is_forward

            # ── TCP handshake: capture initial window sizes ────────────────
            if protocol == 6:
                if raw.tcp_flags & TCPFlags.SYN:
                    if is_forward and not record.fwd_syn_seen:
                        record.init_win_fwd = raw.tcp_window
                        record.fwd_syn_seen = True
                    elif not is_forward and not record.bwd_syn_seen:
                        record.init_win_bwd = raw.tcp_window
                        record.bwd_syn_seen = True

            # ── Append packet ─────────────────────────────────────────────
            record.packets.append(raw)
            record.last_time = now

            # ── Immediate expiry on TCP FIN or RST ────────────────────────
            if protocol == 6 and (
                raw.tcp_flags & TCPFlags.FIN or
                raw.tcp_flags & TCPFlags.RST
            ):
                self._expire(key, record)
                return

            # ── Active timeout (checked on-packet as a fast path) ─────────
            if (now - record.start_time) >= self._config.active_timeout:
                self._expire(key, record)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sweep_loop(self) -> None:
        """Background sweeper — runs every sweeper_interval seconds."""
        while self._running:
            time.sleep(self._config.sweeper_interval)
            self._sweep()

    def _sweep(self) -> None:
        """Evict all flows that have exceeded idle or active timeouts."""
        now = time.time()
        to_expire: List[FlowKey] = []

        with self._lock:
            for key, record in self._table.items():
                idle_s   = now - record.last_time
                active_s = now - record.start_time
                if (
                    idle_s   >= self._config.idle_timeout or
                    active_s >= self._config.active_timeout
                ):
                    to_expire.append(key)

            for key in to_expire:
                if key in self._table:
                    self._expire(key, self._table[key])

            if to_expire:
                logger.debug("Sweeper evicted %d flow(s).", len(to_expire))

    def _expire(self, key: FlowKey, record: FlowRecord) -> None:
        """
        Remove a flow from the table and invoke on_flow_complete.
        Must be called while holding self._lock.
        """
        self._table.pop(key, None)
        if not record.packets:
            return
        try:
            self._on_flow_complete(record)
        except Exception:
            logger.exception(
                "Error in on_flow_complete callback for flow %s:%d → %s:%d.",
                record.src_ip, record.src_port,
                record.dst_ip, record.dst_port,
            )

    def _flush_all(self) -> None:
        """Expire every remaining flow (used on shutdown)."""
        with self._lock:
            keys = list(self._table.keys())
            for key in keys:
                if key in self._table:
                    self._expire(key, self._table[key])

    @property
    def active_flow_count(self) -> int:
        """Number of flows currently tracked (thread-safe snapshot)."""
        with self._lock:
            return len(self._table)


# ─────────────────────────────────────────────────────────────────────────────
# PacketCapture — top-level engine
# ─────────────────────────────────────────────────────────────────────────────

class PacketCapture:
    """
    Orchestrates the full network-engine pipeline:
      packet acquisition → flow aggregation → feature extraction → Redis publish.

    Supports two acquisition modes:
    - **Live**: Scapy ``sniff()`` on a real NIC (requires root / CAP_NET_RAW).
    - **Offline**: ``rdpcap()`` replay of a .pcap file (no privileges needed).

    Usage
    -----
    >>> config = CaptureConfig(interface="eth0", sensor_id="dmz-01")
    >>> engine = PacketCapture(config)
    >>> engine.start()   # blocks; Ctrl-C for graceful shutdown
    """

    def __init__(self, config: CaptureConfig) -> None:
        self._config     = config
        self._extractor  = FlowFeatureExtractor()
        self._redis:     Optional[redis_lib.Redis] = None
        self._aggregator = FlowAggregator(config, self._on_flow_complete)
        self._pkts_seen  = 0
        self._flows_published = 0

    # ── Redis connection ──────────────────────────────────────────────────────

    def _connect_redis(self) -> None:
        self._redis = redis_lib.from_url(
            self._config.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self._redis.ping()
        logger.info(
            "Connected to Redis %s  key=%s",
            self._config.redis_url,
            self._config.redis_key,
        )

    # ── Flow completion callback ───────────────────────────────────────────────

    def _on_flow_complete(self, record: FlowRecord) -> None:
        """
        Called by FlowAggregator when a flow is finalised.
        Extracts features and LPUSH's the JSON payload to Redis.
        """
        try:
            vector = self._extractor.extract(record)
            payload = vector.to_redis_payload()

            if self._redis is not None:
                self._redis.lpush(self._config.redis_key, payload)
                self._flows_published += 1
                logger.debug(
                    "[flow published] %s:%d → %s:%d  proto=%d  pkts=%d  #%d",
                    record.src_ip, record.src_port,
                    record.dst_ip, record.dst_port,
                    record.protocol, len(record.packets),
                    self._flows_published,
                )
        except Exception:
            logger.exception(
                "Failed to extract / publish flow %s:%d → %s:%d.",
                record.src_ip, record.src_port,
                record.dst_ip, record.dst_port,
            )

    # ── Packet callback ───────────────────────────────────────────────────────

    def _on_packet(self, pkt: Packet) -> None:
        self._pkts_seen += 1
        self._aggregator.ingest(pkt)

    def _stop_filter(self, _pkt: Packet) -> bool:
        """Return True to stop sniff() (honoured only when max_packets is set)."""
        if self._config.max_packets is not None:
            return self._pkts_seen >= self._config.max_packets
        return False

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the capture engine.

        Blocks until one of:
        - ``config.max_packets`` is reached (if set)
        - KeyboardInterrupt / SIGTERM
        - All packets consumed (offline PCAP mode)

        After returning, call stop() if you need a clean shutdown from
        an external thread; start() calls stop() internally on completion.
        """
        self._connect_redis()
        self._aggregator.start()

        try:
            if self._config.pcap_file:
                self._run_offline()
            else:
                self._run_live()
        except KeyboardInterrupt:
            logger.info("Capture interrupted by user (KeyboardInterrupt).")
        finally:
            self.stop()

        logger.info(
            "Capture complete — packets: %d  flows published: %d",
            self._pkts_seen, self._flows_published,
        )

    def stop(self) -> None:
        """
        Graceful shutdown: flush remaining flows and disconnect.
        Safe to call multiple times.
        """
        self._aggregator.stop()

    # ── Acquisition modes ─────────────────────────────────────────────────────

    def _run_offline(self) -> None:
        """Replay a PCAP file through the aggregator (no root required)."""
        logger.info("Offline mode — reading %s …", self._config.pcap_file)
        packets = rdpcap(self._config.pcap_file)
        for pkt in packets:
            self._on_packet(pkt)
            if self._stop_filter(pkt):
                break
        # Let the aggregator flush via stop()

    def _run_live(self) -> None:
        """Live capture via Scapy sniff() (requires root / CAP_NET_RAW)."""
        logger.info(
            "Live capture — interface=%s  filter=%r …",
            self._config.interface or "<all>",
            self._config.bpf_filter,
        )
        sniff(
            iface       = self._config.interface,
            filter      = self._config.bpf_filter,
            prn         = self._on_packet,
            stop_filter = self._stop_filter,
            store       = False,    # do not accumulate packets in RAM
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Runtime statistics snapshot (thread-safe)."""
        return {
            "packets_seen":    self._pkts_seen,
            "flows_published": self._flows_published,
            "active_flows":    self._aggregator.active_flow_count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="packet_capture",
        description="Enterprise NIDS — Network Engine packet capture & flow publisher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--interface",  default=None,
                   help="Network interface to capture on (None = all).")
    p.add_argument("--pcap",            default=None,
                   help="Offline PCAP file path (skips live capture).")
    p.add_argument("--filter",          default="ip",
                   help="BPF filter expression.")
    p.add_argument("--redis-url",       default="redis://localhost:6379/0",
                   help="Redis connection URL.")
    p.add_argument("--redis-key",       default="flow:queue",
                   help="Redis list key for flow payloads.")
    p.add_argument("--sensor-id",       default=None,
                   help="Sensor identifier embedded in each FlowFeatureVector.")
    p.add_argument("--segment",         default=None,
                   help="Network segment label (e.g. DMZ, CORP-LAN).")
    p.add_argument("--active-timeout",  type=float, default=120.0,
                   help="Active flow timeout in seconds.")
    p.add_argument("--idle-timeout",    type=float, default=5.0,
                   help="Idle flow timeout in seconds.")
    p.add_argument("--max-packets",     type=int,   default=None,
                   help="Stop after this many packets (default: infinite).")
    p.add_argument("--log-level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   default="INFO")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=getattr(logging, args.log_level),
    )

    config = CaptureConfig(
        interface       = args.interface,
        pcap_file       = args.pcap,
        bpf_filter      = args.filter,
        redis_url       = args.redis_url,
        redis_key       = args.redis_key,
        sensor_id       = args.sensor_id,
        network_segment = args.segment,
        active_timeout  = args.active_timeout,
        idle_timeout    = args.idle_timeout,
        max_packets     = args.max_packets,
    )

    engine = PacketCapture(config)
    engine.start()


if __name__ == "__main__":
    main()
