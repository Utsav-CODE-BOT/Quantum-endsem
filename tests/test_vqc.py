"""Unit and integration tests for ConfigurableVQC training, prediction, and reproducibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.classification_metrics import compute_classification_metrics
from src.quantum.vqc_model import ConfigurableVQC


class TestConfigurableVQC:
    """Test VQC training, prediction, and reproducibility."""

    @pytest.fixture
    def synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Small linearly separable 4-feature dataset for fast deterministic testing."""
        X = np.array([
            [0.1, 0.2, 0.1, 0.2],
            [0.2, 0.1, 0.2, 0.1],
            [0.15, 0.25, 0.1, 0.2],
            [2.8, 2.9, 3.0, 2.8],
            [2.9, 3.0, 2.9, 3.0],
            [3.0, 2.8, 2.8, 2.9],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y

    def test_vqc_training_and_prediction(self, synthetic_data) -> None:
        X, y = synthetic_data
        vqc = ConfigurableVQC(
            num_qubits=4,
            num_layers=1,
            rotation_gates=["ry"],
            entanglement_type="linear",
            optimizer_name="cobyla",
            maxiter=15,
            random_seed=42,
        )

        assert not vqc.is_fitted
        vqc.fit(X, y)
        assert vqc.is_fitted
        assert vqc.training_history.total_evaluations > 0
        assert len(vqc.training_history.loss_history) > 0

        preds = vqc.predict(X)
        assert preds.shape == (len(X),)
        assert all(p in [0, 1] for p in preds)

        probas = vqc.predict_proba(X)
        assert probas.shape == (len(X), 2)
        # Probabilities sum to 1.0 within tolerance
        np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

    def test_vqc_reproducibility(self, synthetic_data) -> None:
        """Verify that identical random seeds produce identical trained weights."""
        X, y = synthetic_data

        vqc1 = ConfigurableVQC(
            num_qubits=4,
            num_layers=1,
            rotation_gates=["ry"],
            entanglement_type="linear",
            optimizer_name="cobyla",
            maxiter=10,
            random_seed=123,
        )
        vqc1.fit(X, y)

        vqc2 = ConfigurableVQC(
            num_qubits=4,
            num_layers=1,
            rotation_gates=["ry"],
            entanglement_type="linear",
            optimizer_name="cobyla",
            maxiter=10,
            random_seed=123,
        )
        vqc2.fit(X, y)

        np.testing.assert_allclose(vqc1.weights, vqc2.weights, atol=1e-5)

    def test_vqc_checkpointing(self, synthetic_data, tmp_path: Path) -> None:
        X, y = synthetic_data
        vqc = ConfigurableVQC(
            num_qubits=4,
            num_layers=1,
            rotation_gates=["ry"],
            optimizer_name="cobyla",
            maxiter=10,
            random_seed=42,
        )
        vqc.fit(X, y)

        ckpt_path = tmp_path / "vqc_checkpoint.pkl"
        vqc.save(ckpt_path)

        data = ConfigurableVQC.load_checkpoint(ckpt_path)
        assert data["num_qubits"] == 4
        assert data["weights"] is not None
        assert "metadata" in data
        assert "training_history" in data

    def test_metrics_integration(self, synthetic_data) -> None:
        X, y = synthetic_data
        vqc = ConfigurableVQC(
            num_qubits=4,
            num_layers=1,
            maxiter=10,
            random_seed=42,
        )
        vqc.fit(X, y)

        preds = vqc.predict(X)
        probs = vqc.predict_proba(X)
        report = compute_classification_metrics(y, preds, probs)

        assert 0.0 <= report.accuracy <= 1.0
        assert 0.0 <= report.f1_macro <= 1.0
        assert len(report.confusion_matrix) == 2
