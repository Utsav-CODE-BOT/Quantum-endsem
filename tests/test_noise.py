"""Tests for noise simulation and noise-aware evaluation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.noise_evaluator import (
    NoiseEvaluationReport,
    NoiseEvaluator,
    NoiseModelBuilder,
    NoiseModelConfig,
)
from src.quantum.ansatz_builder import AnsatzBuilder
from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.feature_encoding import AngleFeatureMap
from src.quantum.vqc_model import ConfigurableVQC


@pytest.fixture
def dummy_data() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    # 20 samples, 4 features
    X = np.random.uniform(0.0, np.pi, size=(20, 4))
    y = np.random.choice([0, 1], size=20)
    return X, y


@pytest.fixture
def trained_dummy_vqc(dummy_data: tuple[np.ndarray, np.ndarray]) -> ConfigurableVQC:
    X, y = dummy_data
    vqc = ConfigurableVQC(
        num_qubits=4,
        num_layers=1,
        rotation_gates=["ry"],
        entanglement_type="linear",
        maxiter=5,
        random_seed=42,
    )
    vqc.fit(X[:10], y[:10])
    return vqc


def test_noise_model_builder_construction():
    """Verify that NoiseModelBuilder creates valid Qiskit Aer NoiseModel with proper errors."""
    config = NoiseModelConfig(
        name="test_config",
        single_qubit_error=0.002,
        two_qubit_error=0.015,
        measurement_error=0.02,
        shots=512,
    )
    model = NoiseModelBuilder.build(config)
    assert model is not None
    # Basic inspection of basis gates added
    assert "cx" in model.basis_gates or len(model.basis_gates) > 0


def test_noise_evaluator_unfitted_raises_error(dummy_data: tuple[np.ndarray, np.ndarray]):
    """Verify that attempting to evaluate an unfitted VQC raises RuntimeError."""
    X, y = dummy_data
    vqc = ConfigurableVQC(
        num_qubits=4,
        num_layers=1,
        rotation_gates=["ry"],
        entanglement_type="linear",
        random_seed=42,
    )
    evaluator = NoiseEvaluator(seed=42)
    config = NoiseModelConfig(name="low", single_qubit_error=0.001)

    with pytest.raises(RuntimeError, match="VQC must be fitted"):
        evaluator.evaluate(vqc, X, y, config)


def test_noise_evaluator_runs_and_reports(trained_dummy_vqc: ConfigurableVQC, dummy_data: tuple[np.ndarray, np.ndarray]):
    """Verify paired ideal vs noisy evaluation yields valid reports and retention metrics."""
    X, y = dummy_data
    X_test, y_test = X[10:], y[10:]
    evaluator = NoiseEvaluator(seed=42)
    config = NoiseModelConfig(
        name="low_test",
        single_qubit_error=0.001,
        two_qubit_error=0.01,
        measurement_error=0.01,
        shots=256,
    )

    report = evaluator.evaluate(trained_dummy_vqc, X_test, y_test, config)

    assert isinstance(report, NoiseEvaluationReport)
    assert 0.0 <= report.ideal_metrics.accuracy <= 1.0
    assert 0.0 <= report.noisy_metrics.accuracy <= 1.0
    assert -1.0 <= report.accuracy_drop <= 1.0
    assert report.relative_accuracy_retention >= 0.0
    assert report.noise_sensitivity_index >= 0.0

    report_dict = report.to_dict()
    assert "arch_id" in report_dict
    assert "ideal_metrics" in report_dict
    assert "noisy_metrics" in report_dict
    assert "relative_accuracy_retention" in report_dict
