"""VQC circuit generator combining quantum feature encoding and variational ansatz.

Extracts complete quantum circuit resource metrics:
- Qubit count
- Decomposed circuit depth
- Total gate count and 1-qubit vs 2-qubit breakdown
- Gate type distribution
- Parameter counts (input features vs trainable weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from qiskit import QuantumCircuit

from src.quantum.ansatz_builder import AnsatzBuilder
from src.quantum.feature_encoding import AngleFeatureMap
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CircuitMetadata:
    """Quantitative structural and resource metrics of a quantum circuit."""

    num_qubits: int
    circuit_depth: int
    total_gates: int
    single_qubit_gates: int
    two_qubit_gates: int
    gate_distribution: dict[str, int]
    num_input_features: int
    num_trainable_parameters: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for reporting and fitness calculation."""
        return {
            "num_qubits": self.num_qubits,
            "circuit_depth": self.circuit_depth,
            "total_gates": self.total_gates,
            "single_qubit_gates": self.single_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "gate_distribution": self.gate_distribution,
            "num_input_features": self.num_input_features,
            "num_trainable_parameters": self.num_trainable_parameters,
        }


class VQCCircuitGenerator:
    """Generates combined VQC circuits from feature maps and variational ansatze."""

    def __init__(
        self,
        feature_map: AngleFeatureMap,
        ansatz: AnsatzBuilder,
    ) -> None:
        """Initialize generator.

        Args:
            feature_map: AngleFeatureMap instance.
            ansatz: AnsatzBuilder instance.
        """
        if feature_map.num_qubits != ansatz.num_qubits:
            raise ValueError(
                f"Feature map qubits ({feature_map.num_qubits}) must match ansatz qubits ({ansatz.num_qubits})"
            )

        self.feature_map = feature_map
        self.ansatz = ansatz
        self.num_qubits = feature_map.num_qubits

        # Combined full circuit: feature_map + ansatz
        self.full_circuit = self._compose_circuit()
        self.metadata = self.extract_metadata()

    def _compose_circuit(self) -> QuantumCircuit:
        """Compose feature map and variational ansatz into a single circuit."""
        qc = QuantumCircuit(self.num_qubits, name="VQCCircuit")
        qc.compose(self.feature_map.circuit, inplace=True)
        qc.compose(self.ansatz.circuit, inplace=True)
        return qc

    def extract_metadata(self) -> CircuitMetadata:
        """Compute circuit metrics on the decomposed circuit representation."""
        # Decompose custom composites to count standard basis gates
        decomposed = self.full_circuit.decompose()
        gate_counts = decomposed.count_ops()

        total_gates = sum(gate_counts.values())
        two_qubit_gates = gate_counts.get("cx", 0) + gate_counts.get("cz", 0)
        single_qubit_gates = total_gates - two_qubit_gates

        return CircuitMetadata(
            num_qubits=self.num_qubits,
            circuit_depth=decomposed.depth(),
            total_gates=total_gates,
            single_qubit_gates=single_qubit_gates,
            two_qubit_gates=two_qubit_gates,
            gate_distribution=dict(gate_counts),
            num_input_features=self.feature_map.num_parameters,
            num_trainable_parameters=self.ansatz.num_parameters,
        )

    @classmethod
    def from_specs(
        cls,
        num_qubits: int,
        num_layers: int = 2,
        rotation_gates: Sequence[str] = ("ry", "rz"),
        entanglement_type: str = "linear",
        entanglement_freq: int = 1,
        feature_rotation: str = "ry",
    ) -> "VQCCircuitGenerator":
        """Factory method to build generator directly from architectural hyperparameters."""
        feature_map = AngleFeatureMap(num_qubits=num_qubits, rotation_gate=feature_rotation)
        ansatz = AnsatzBuilder(
            num_qubits=num_qubits,
            num_layers=num_layers,
            rotation_gates=rotation_gates,
            entanglement_type=entanglement_type,
            entanglement_freq=entanglement_freq,
        )
        return cls(feature_map=feature_map, ansatz=ansatz)
