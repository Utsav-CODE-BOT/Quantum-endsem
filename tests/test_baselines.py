"""Unit tests for classical and fixed quantum baseline models."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.classical_nn import ClassicalMLP
from src.baselines.classical_svm import ClassicalSVM
from src.baselines.fixed_vqc import (
    create_fixed_deep_vqc,
    create_fixed_shallow_vqc,
    create_fixed_standard_vqc,
)


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable synthetic dataset for baseline testing."""
    np.random.seed(42)
    X = np.random.uniform(0.0, 3.0, size=(30, 4))
    y = (X[:, 0] + X[:, 1] > 3.0).astype(int)
    return X, y


def test_classical_svm_linear_and_rbf(synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = synthetic_data
    X_train, y_train = X[:20], y[:20]
    X_test, y_test = X[20:], y[20:]

    for kernel in ("linear", "rbf"):
        svm = ClassicalSVM(kernel=kernel, C=1.0, random_seed=42)
        assert not svm.is_fitted
        svm.fit(X_train, y_train)
        assert svm.is_fitted
        assert svm.num_parameters > 0

        preds = svm.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(preds).issubset({0, 1})

        probs = svm.predict_proba(X_test)
        assert probs.shape == (len(X_test), 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

        report = svm.evaluate(X_test, y_test)
        assert 0.0 <= report.accuracy <= 1.0


def test_classical_mlp(synthetic_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = synthetic_data
    X_train, y_train = X[:20], y[:20]
    X_test, y_test = X[20:], y[20:]

    mlp = ClassicalMLP(hidden_layer_sizes=(4,), max_iter=50, random_seed=42)
    assert not mlp.is_fitted
    mlp.fit(X_train, y_train)
    assert mlp.is_fitted
    assert mlp.num_parameters > 0

    preds = mlp.predict(X_test)
    assert preds.shape == (len(X_test),)

    probs = mlp.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    report = mlp.evaluate(X_test, y_test)
    assert 0.0 <= report.accuracy <= 1.0


def test_fixed_vqc_factories() -> None:
    shallow = create_fixed_shallow_vqc(num_qubits=4, maxiter=5)
    assert shallow.num_layers == 1
    assert shallow.metadata.circuit_depth > 0
    assert shallow.metadata.num_trainable_parameters == 8

    standard = create_fixed_standard_vqc(num_qubits=4, maxiter=5)
    assert standard.num_layers == 2
    assert standard.metadata.num_trainable_parameters == 24

    deep = create_fixed_deep_vqc(num_qubits=4, maxiter=5)
    assert deep.num_layers == 4
    assert deep.metadata.num_trainable_parameters == 40
    assert deep.metadata.circuit_depth > standard.metadata.circuit_depth
