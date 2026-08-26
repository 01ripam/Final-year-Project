"""
test_inference_worker.py
────────────────────────
Unit tests for the ML inference pipeline components.

Tests the model interface, dummy classifier, model registry, and
the inference worker logic — all without requiring Redis, Celery,
or TimescaleDB.

Coverage
────────
  - BaseClassifier / DummyClassifier prediction shape & value ranges
  - DummyClassifier determinism (same input → same output)
  - ModelRegistry register / get / set_active / unregister
  - DummySHAPExplainer output structure
  - Inference pipeline logic (mocked dependencies)
  - FlowPrediction output validation against flow_schema contracts

Run
───
    pytest tests/unit/test_inference_worker.py -v
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, ".")
sys.path.insert(0, "src/data")
sys.path.insert(0, "src/models")
sys.path.insert(0, "src/workers")
sys.path.insert(0, "src/network_engine")

from src.models.base import BaseClassifier, CLASS_NAMES, NUM_CLASSES
from src.models.dummy import DummyClassifier, DEFAULT_FEATURE_COLUMNS
from src.models.registry import ModelRegistry
from src.data.flow_schema import (
    FlowFeatureVector,
    FlowPrediction,
    AttackClass,
    ShapExplanation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_vector_kwargs() -> dict:
    """
    Build kwargs for a valid FlowFeatureVector — reuses the factory from
    test_flow_schema.py but defined locally so this test file is self-contained.
    """
    return dict(
        src_ip="192.168.1.10", dst_ip="8.8.8.8",
        src_port=54321, dst_port=443, protocol=6,
        flow_duration=1_000_000.0,
        total_fwd_packets=10, total_bwd_packets=8,
        total_length_fwd_packets=1400.0, total_length_bwd_packets=800.0,
        fwd_packet_length_max=140.0, fwd_packet_length_min=20.0,
        fwd_packet_length_mean=70.0, fwd_packet_length_std=30.0,
        bwd_packet_length_max=100.0, bwd_packet_length_min=20.0,
        bwd_packet_length_mean=50.0, bwd_packet_length_std=20.0,
        flow_bytes_per_sec=2200.0, flow_packets_per_sec=18.0,
        flow_iat_mean=55_000.0, flow_iat_std=5_000.0,
        flow_iat_max=200_000.0, flow_iat_min=1_000.0,
        fwd_iat_total=900_000.0, fwd_iat_mean=100_000.0,
        fwd_iat_std=10_000.0, fwd_iat_max=300_000.0, fwd_iat_min=5_000.0,
        bwd_iat_total=700_000.0, bwd_iat_mean=87_500.0,
        bwd_iat_std=8_000.0, bwd_iat_max=250_000.0, bwd_iat_min=4_000.0,
        fwd_psh_flags=1, bwd_psh_flags=0, fwd_urg_flags=0, bwd_urg_flags=0,
        fin_flag_count=1, syn_flag_count=1, rst_flag_count=0,
        psh_flag_count=1, ack_flag_count=12, urg_flag_count=0,
        cwe_flag_count=0, ece_flag_count=0,
        fwd_header_length=200, bwd_header_length=160,
        fwd_packets_per_sec=10.0, bwd_packets_per_sec=8.0,
        min_packet_length=20.0, max_packet_length=140.0,
        packet_length_mean=62.0, packet_length_std=25.0,
        packet_length_variance=625.0,
        active_mean=500_000.0, active_std=50_000.0,
        active_max=800_000.0, active_min=200_000.0,
        idle_mean=100_000.0, idle_std=10_000.0,
        idle_max=300_000.0, idle_min=10_000.0,
        subflow_fwd_packets=10, subflow_fwd_bytes=1400.0,
        subflow_bwd_packets=8, subflow_bwd_bytes=800.0,
        init_win_bytes_fwd=65535, init_win_bytes_bwd=65535,
        act_data_pkt_fwd=9, min_seg_size_fwd=20,
        down_up_ratio=0.8, avg_packet_size=62.0,
        avg_fwd_segment_size=70.0, avg_bwd_segment_size=50.0,
        fwd_avg_bytes_per_bulk=0.0, fwd_avg_packets_per_bulk=0.0,
        fwd_avg_bulk_rate=0.0, bwd_avg_bytes_per_bulk=0.0,
        bwd_avg_packets_per_bulk=0.0, bwd_avg_bulk_rate=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DummyClassifier Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDummyClassifier:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model = DummyClassifier()
        self.features = np.random.rand(1, len(DEFAULT_FEATURE_COLUMNS))

    def test_predict_proba_shape(self):
        proba = self.model.predict_proba(self.features)
        assert proba.shape == (1, NUM_CLASSES)

    def test_predict_proba_sums_to_one(self):
        proba = self.model.predict_proba(self.features)
        assert proba[0].sum() == pytest.approx(1.0, abs=1e-6)

    def test_predict_proba_non_negative(self):
        proba = self.model.predict_proba(self.features)
        assert (proba >= 0.0).all()

    def test_predict_returns_valid_class_idx(self):
        idx, conf, probs = self.model.predict(self.features)
        assert 0 <= idx < NUM_CLASSES
        assert 0.0 <= conf <= 1.0
        assert len(probs) == NUM_CLASSES

    def test_predict_class_name_in_probs(self):
        _, _, probs = self.model.predict(self.features)
        for cls_name in CLASS_NAMES:
            assert cls_name in probs

    def test_deterministic_same_input(self):
        features = np.ones((1, len(DEFAULT_FEATURE_COLUMNS))) * 42.0
        r1 = self.model.predict_proba(features)
        r2 = self.model.predict_proba(features)
        np.testing.assert_array_equal(r1, r2)

    def test_different_input_different_output(self):
        f1 = np.ones((1, len(DEFAULT_FEATURE_COLUMNS))) * 1.0
        f2 = np.ones((1, len(DEFAULT_FEATURE_COLUMNS))) * 99.0
        r1 = self.model.predict_proba(f1)
        r2 = self.model.predict_proba(f2)
        # Extremely unlikely to be identical for different inputs
        assert not np.allclose(r1, r2)

    def test_batch_prediction(self):
        batch = np.random.rand(5, len(DEFAULT_FEATURE_COLUMNS))
        results = self.model.predict_batch(batch)
        assert len(results) == 5
        for idx, conf, probs in results:
            assert 0 <= idx < NUM_CLASSES
            assert 0.0 <= conf <= 1.0

    def test_1d_input_handled(self):
        features_1d = np.random.rand(len(DEFAULT_FEATURE_COLUMNS))
        idx, conf, probs = self.model.predict(features_1d)
        assert 0 <= idx < NUM_CLASSES

    def test_feature_columns_list(self):
        assert len(self.model.feature_columns) == 76
        assert self.model.feature_columns[0] == "flow_duration"

    def test_name(self):
        assert self.model.name == "dummy-v1.0"


# ─────────────────────────────────────────────────────────────────────────────
# ModelRegistry Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRegistry:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Fresh registry for each test (not the module singleton)
        self.reg = ModelRegistry()

    def test_register_and_get(self):
        model = DummyClassifier()
        self.reg.register("test-model", model)
        assert self.reg.get("test-model") is model

    def test_auto_activate_first_model(self):
        model = DummyClassifier()
        self.reg.register("first", model)
        assert self.reg.active_name == "first"
        assert self.reg.active is model

    def test_set_active(self):
        m1 = DummyClassifier()
        m2 = DummyClassifier()
        self.reg.register("m1", m1)
        self.reg.register("m2", m2)
        self.reg.set_active("m2")
        assert self.reg.active is m2

    def test_set_active_unknown_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            self.reg.set_active("nonexistent")

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            self.reg.get("nonexistent")

    def test_active_when_empty_raises(self):
        with pytest.raises(RuntimeError, match="No active model"):
            _ = self.reg.active

    def test_unregister(self):
        model = DummyClassifier()
        self.reg.register("tmp", model)
        self.reg.unregister("tmp")
        assert "tmp" not in self.reg.available

    def test_unregister_active_clears_active(self):
        model = DummyClassifier()
        self.reg.register("only", model)
        self.reg.unregister("only")
        assert self.reg.active_name is None

    def test_available_lists_all(self):
        self.reg.register("a", DummyClassifier())
        self.reg.register("b", DummyClassifier())
        assert sorted(self.reg.available) == ["a", "b"]

    def test_len(self):
        assert len(self.reg) == 0
        self.reg.register("x", DummyClassifier())
        assert len(self.reg) == 1

    def test_replace_model(self):
        m1 = DummyClassifier()
        m2 = DummyClassifier()
        self.reg.register("model", m1)
        self.reg.register("model", m2)
        assert self.reg.get("model") is m2


# ─────────────────────────────────────────────────────────────────────────────
# DummySHAPExplainer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDummySHAPExplainer:
    def test_explain_returns_correct_structure(self):
        from src.workers.shap_worker import DummySHAPExplainer

        model = DummyClassifier()
        features = np.random.rand(len(DEFAULT_FEATURE_COLUMNS))
        explainer = DummySHAPExplainer()

        shap_values, base_value, top_features = explainer.explain(
            model, features, model.feature_columns,
        )

        assert isinstance(shap_values, dict)
        assert len(shap_values) == len(model.feature_columns)
        assert isinstance(base_value, float)
        assert isinstance(top_features, list)
        assert len(top_features) <= 10

    def test_explain_deterministic(self):
        from src.workers.shap_worker import DummySHAPExplainer

        model = DummyClassifier()
        features = np.ones(len(DEFAULT_FEATURE_COLUMNS)) * 5.0
        explainer = DummySHAPExplainer()

        sv1, bv1, tf1 = explainer.explain(model, features, model.feature_columns)
        sv2, bv2, tf2 = explainer.explain(model, features, model.feature_columns)

        assert sv1 == sv2
        assert bv1 == bv2
        assert tf1 == tf2

    def test_explainer_type_string(self):
        from src.workers.shap_worker import DummySHAPExplainer
        assert DummySHAPExplainer().explainer_type == "DummyExplainer"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Inference Logic Tests (no Celery / Redis required)
# ─────────────────────────────────────────────────────────────────────────────

class TestInferencePipelineLogic:
    """
    Tests the core inference logic — model prediction + FlowPrediction
    construction — without requiring Celery or Redis.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model = DummyClassifier()
        self.vector = FlowFeatureVector(**_minimal_vector_kwargs())

    def test_feature_array_extraction(self):
        arr = self.vector.to_feature_array(self.model.feature_columns)
        assert len(arr) == len(self.model.feature_columns)
        # First feature is flow_duration
        assert arr[0] == self.vector.flow_duration

    def test_inference_produces_valid_prediction(self):
        features = np.array(
            self.vector.to_feature_array(self.model.feature_columns),
            dtype=np.float64,
        ).reshape(1, -1)

        class_idx, confidence, class_probs = self.model.predict(features)
        predicted_class = CLASS_NAMES[class_idx]

        prediction = FlowPrediction(
            flow_id=self.vector.flow_id,
            predicted_class=AttackClass(predicted_class),
            confidence=confidence,
            class_probabilities=class_probs,
            model_name=self.model.name,
            inference_latency_ms=1.23,
            shap_queued=(predicted_class != "Normal"),
        )

        assert prediction.flow_id == self.vector.flow_id
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.model_name == "dummy-v1.0"
        assert len(prediction.class_probabilities) == NUM_CLASSES

    def test_prediction_json_roundtrip(self):
        features = np.array(
            self.vector.to_feature_array(self.model.feature_columns),
            dtype=np.float64,
        ).reshape(1, -1)

        class_idx, confidence, class_probs = self.model.predict(features)
        predicted_class = CLASS_NAMES[class_idx]

        prediction = FlowPrediction(
            flow_id=self.vector.flow_id,
            predicted_class=AttackClass(predicted_class),
            confidence=confidence,
            class_probabilities=class_probs,
            model_name=self.model.name,
            inference_latency_ms=5.0,
        )

        # Serialise + deserialise
        prediction_json = prediction.model_dump_json()
        restored = FlowPrediction.model_validate_json(prediction_json)

        assert restored.flow_id == prediction.flow_id
        assert restored.predicted_class == prediction.predicted_class
        assert restored.confidence == pytest.approx(prediction.confidence)

    def test_shap_explanation_construction(self):
        from src.workers.shap_worker import DummySHAPExplainer

        features = np.array(
            self.vector.to_feature_array(self.model.feature_columns),
            dtype=np.float64,
        )

        explainer = DummySHAPExplainer()
        shap_values, base_value, top_features = explainer.explain(
            self.model, features, self.model.feature_columns,
        )

        explanation = ShapExplanation(
            flow_id=self.vector.flow_id,
            shap_values=shap_values,
            base_value=base_value,
            top_features=top_features,
            explainer_type=explainer.explainer_type,
            explanation_latency_ms=10.0,
        )

        assert explanation.flow_id == self.vector.flow_id
        assert len(explanation.shap_values) == len(self.model.feature_columns)
        assert len(explanation.top_features) <= 10

        # JSON round-trip
        expl_json = explanation.model_dump_json()
        restored = ShapExplanation.model_validate_json(expl_json)
        assert restored.flow_id == explanation.flow_id

    def test_all_attack_classes_are_valid_enum_values(self):
        """Ensure CLASS_NAMES from base.py align with AttackClass enum."""
        for name in CLASS_NAMES:
            assert AttackClass(name) is not None


class TestFeatureColumnAlignment:
    """
    Verify that DummyClassifier.feature_columns matches FlowFeatureVector
    fields — critical for to_feature_array() to work correctly.
    """

    def test_all_feature_columns_exist_in_vector(self):
        vector = FlowFeatureVector(**_minimal_vector_kwargs())
        data = vector.model_dump()
        model = DummyClassifier()

        for col in model.feature_columns:
            assert col in data, f"Feature column {col!r} not found in FlowFeatureVector"

    def test_feature_array_length_matches(self):
        vector = FlowFeatureVector(**_minimal_vector_kwargs())
        model = DummyClassifier()
        arr = vector.to_feature_array(model.feature_columns)
        assert len(arr) == len(model.feature_columns)
