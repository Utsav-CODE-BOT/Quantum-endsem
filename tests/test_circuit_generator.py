"""Unit tests for AnsatzBuilder and VQCCircuitGenerator."""

from __future__ import annotations

import numpy as np
import pytest

from src.quantum.ansatz_builder import AnsatzBuilder
from src.quantum.circuit_generator import CircuitMetadata, VQCCircuitGenerator
from src.quantum.feature_encoding import AngleFeatureMap
from src.quantum.measurement import MeasurementStrategy, parity_interpret, single_qubit_interpret


class TestAnsatzBuilder:
    """Test ansatz construction, depth, and parameter counts across topologies."""

    def test_ansatz_linear_entanglement(self) -> None:
        num_qubits = 4
        num_layers = 2
        rotations = ["ry", "rz"]
        builder = AnsatzBuilder(
            num_qubits=num_qubits,
            num_layers=num_layers,
            rotation_gates=rotations,
            entanglement_type="linear",
        )
        # Expected parameters: (layers + 1) * qubits * len(rotations)
        # (2 + 1) * 4 * 2 = 24 parameters
        assert builder.num_parameters == 24
        assert builder.circuit.num_qubits == num_qubits

    def test_ansatz_circular_entanglement(self) -> None:
        builder = AnsatzBuilder(
            num_qubits=4,
            num_layers=1,
            rotation_gates=["ry"],
            entanglement_type="circular",
        )
        # (1 + 1) * 4 * 1 = 8 parameters
        assert builder.num_parameters == 8
        ops = builder.circuit.decompose().count_ops()
        # Circular entanglement on 4 qubits has 4 CX gates per layer
        assert ops.get("cx", 0) == 4

    def test_ansatz_pairwise_entanglement(self) -> None:
        builder = AnsatzBuilder(
            num_qubits=4,
            num_layers=2,
            rotation_gates=["ry"],
            entanglement_type="pairwise",
        )
        assert builder.num_parameters == 12
        assert builder.circuit.num_qubits == 4

    def test_ansatz_none_entanglement(self) -> None:
        builder = AnsatzBuilder(
            num_qubits=3,
            num_layers=2,
            rotation_gates=["rx"],
            entanglement_type="none",
        )
        ops = builder.circuit.decompose().count_ops()
        assert "cx" not in ops


class TestVQCCircuitGenerator:
    """Test full VQC circuit composition and metadata extraction."""

    def test_circuit_composition_and_metadata(self) -> None:
        gen = VQCCircuitGenerator.from_specs(
            num_qubits=4,
            num_layers=2,
            rotation_gates=["ry", "rz"],
            entanglement_type="linear",
            entanglement_freq=1,
            feature_rotation="ry",
        )

        assert isinstance(gen.metadata, CircuitMetadata)
        assert gen.metadata.num_qubits == 4
        assert gen.metadata.num_input_features == 4
        assert gen.metadata.num_trainable_parameters == 24
        assert gen.metadata.circuit_depth > 0
        assert gen.metadata.two_qubit_gates > 0
        assert gen.metadata.single_qubit_gates > 0
        assert "cx" in gen.metadata.gate_distribution

    def test_mismatched_qubits_raises(self) -> None:
        fm = AngleFeatureMap(num_qubits=4)
        ansatz = AnsatzBuilder(num_qubits=6)
        with pytest.raises(ValueError, match="Feature map qubits .* must match ansatz qubits"):
            VQCCircuitGenerator(feature_map=fm, ansatz=ansatz)


class TestMeasurementStrategies:
    """Test bitstring measurement post-processing."""

    def test_parity_interpretation(self) -> None:
        # 0 -> '0' -> 0
        # 1 -> '1' -> 1
        # 2 -> '10' -> 1
        # 3 -> '11' -> 0
        # 7 -> '111' -> 1
        assert parity_interpret(0) == 0
        assert parity_interpret(1) == 1
        assert parity_interpret(2) == 1
        assert parity_interpret(3) == 0
        assert parity_interpret(7) == 1

    def test_single_qubit_interpretation(self) -> None:
        assert single_qubit_interpret(0) == 0
        assert single_qubit_interpret(1) == 1
        assert single_qubit_interpret(2) == 0
        assert single_qubit_interpret(3) == 1

    def test_strategy_factory(self) -> None:
        func = MeasurementStrategy.get_interpret_func("parity")
        assert func(3) == 0
        with pytest.raises(ValueError, match="Unsupported measurement strategy"):
            MeasurementStrategy.get_interpret_func("invalid_strategy")
