"""Unit tests for quantum angle feature encoding."""

from __future__ import annotations

import numpy as np
import pytest

from src.quantum.feature_encoding import AngleFeatureMap


class TestAngleFeatureMap:
    """Test angle feature map circuit construction and parameter binding."""

    def test_construction(self) -> None:
        feature_map = AngleFeatureMap(num_qubits=4, rotation_gate="ry")
        assert feature_map.num_qubits == 4
        assert feature_map.num_parameters == 4
        assert feature_map.circuit.num_qubits == 4
        assert len(feature_map.circuit.parameters) == 4

    def test_parameter_binding(self) -> None:
        feature_map = AngleFeatureMap(num_qubits=4, rotation_gate="ry")
        sample_angles = np.array([0.1, 1.2, 2.3, 3.14])
        bound_qc = feature_map.assign_features(sample_angles)

        assert bound_qc.num_parameters == 0  # Fully bound
        assert bound_qc.depth() == 1  # Parallel 1-qubit rotations

    def test_dimension_mismatch_raises(self) -> None:
        feature_map = AngleFeatureMap(num_qubits=4)
        with pytest.raises(ValueError, match="Expected 1D feature vector"):
            feature_map.assign_features(np.array([0.1, 0.2]))  # Length 2 != 4

    def test_supported_rotations(self) -> None:
        for rot in ["rx", "ry", "rz"]:
            fm = AngleFeatureMap(num_qubits=2, rotation_gate=rot)
            assert rot in fm.circuit.name.lower()

        with pytest.raises(ValueError, match="Unsupported rotation_gate"):
            AngleFeatureMap(num_qubits=2, rotation_gate="h")
