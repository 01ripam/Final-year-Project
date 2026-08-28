"""
dummy.py
────────
Dummy classifier for end-to-end pipeline testing.

Produces deterministic pseudo-random predictions seeded by the flow's
feature values so that results are reproducible across runs.  The
predictions are *not* meaningful — use only for testing the pipeline
wiring (Redis → inference → SHAP → DB).

Author : Person 1 — Data & Classical ML
Project: Enterprise NIDS — Week 2
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from src.models.base import BaseClassifier, CLASS_NAMES, NUM_CLASSES


# ── Default feature column list (78 CICIDS2017 features) ─────────────────────
# This matches the FlowFeatureVector field order, excluding identity/metadata.
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "destination_port",
    "protocol",
    "flow_duration",
    "total_fwd_packets", "total_bwd_packets",
    "total_length_fwd_packets", "total_length_bwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min",
    "fwd_packet_length_mean", "fwd_packet_length_std",
    "bwd_packet_length_max", "bwd_packet_length_min",
    "bwd_packet_length_mean", "bwd_packet_length_std",
    "flow_bytes_per_sec", "flow_packets_per_sec",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std",
    "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std",
    "bwd_iat_max", "bwd_iat_min",
    "fwd_psh_flags", "bwd_psh_flags",
    "fwd_urg_flags", "bwd_urg_flags",
    "fin_flag_count", "syn_flag_count", "rst_flag_count",
    "psh_flag_count", "ack_flag_count", "urg_flag_count",
    "cwe_flag_count", "ece_flag_count",
    "fwd_header_length", "bwd_header_length",
    "fwd_packets_per_sec", "bwd_packets_per_sec",
    "min_packet_length", "max_packet_length",
    "packet_length_mean", "packet_length_std", "packet_length_variance",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
    "subflow_fwd_packets", "subflow_fwd_bytes",
    "subflow_bwd_packets", "subflow_bwd_bytes",
    "init_win_bytes_fwd", "init_win_bytes_bwd",
    "act_data_pkt_fwd", "min_seg_size_fwd",
    "down_up_ratio", "avg_packet_size",
    "avg_fwd_segment_size", "avg_bwd_segment_size",
    "fwd_avg_bytes_per_bulk", "fwd_avg_packets_per_bulk",
    "fwd_avg_bulk_rate",
    "bwd_avg_bytes_per_bulk", "bwd_avg_packets_per_bulk",
    "bwd_avg_bulk_rate",
]


class DummyClassifier(BaseClassifier):
    """
    Deterministic pseudo-random classifier for pipeline testing.

    Algorithm
    ---------
    1. Hash the feature vector bytes → 32-bit seed.
    2. Use NumPy ``RandomState(seed)`` to generate a Dirichlet sample
       over 7 classes (so probabilities sum to 1.0).
    3. Return the Dirichlet sample as ``predict_proba`` output.

    The result is:
    - Deterministic for the same input (same hash → same seed → same sample).
    - Plausibly distributed across all 7 classes (not always "Normal").
    - Fast (no actual ML computation).

    Parameters
    ----------
    attack_bias : float
        Controls how often the dummy produces attack vs Normal predictions.
        Values > 1.0 bias toward attacks; < 1.0 bias toward Normal.
        Default 1.0 = uniform Dirichlet (roughly equal class probability).
    """

    def __init__(self, attack_bias: float = 1.0) -> None:
        self.name = "dummy-v1.0"
        self.feature_columns = list(DEFAULT_FEATURE_COLUMNS)
        self.num_classes = NUM_CLASSES
        self._attack_bias = attack_bias

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Produce deterministic pseudo-random class probabilities.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, F)`` feature matrix.

        Returns
        -------
        np.ndarray
            Shape ``(N, 7)`` probability matrix.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_samples = features.shape[0]
        proba = np.zeros((n_samples, self.num_classes), dtype=np.float64)

        for i in range(n_samples):
            # Hash feature row → deterministic seed
            row_bytes = features[i].tobytes()
            digest = hashlib.md5(row_bytes).hexdigest()
            seed = int(digest[:8], 16) % (2**31)

            rng = np.random.RandomState(seed)

            # Dirichlet alpha: Normal gets 1.0, attacks get attack_bias
            alpha = np.ones(self.num_classes)
            alpha[1:] *= self._attack_bias   # bias attack classes

            proba[i] = rng.dirichlet(alpha)

        return proba
