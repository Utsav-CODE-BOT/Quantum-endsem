"""Quantum feature encoding maps continuous input features into quantum states.

Uses explicit angle encoding where each classical feature x_i rotates qubit i
via a parameterized single-qubit rotation (RY by default).
Input parameters are cleanly represented as a Qiskit ParameterVector('x', n_qubits).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AngleFeatureMap:
    """Constructs a parameterized angle-encoding QuantumCircuit for classical features."""

    SUPPORTED_ROTATIONS = ("rx", "ry", "rz")

    def __init__(
        self,
        num_qubits: int,
        rotation_gate: str = "ry",
        parameter_prefix: str = "x",
    ) -> None:
        """Initialize the angle feature map.

        Args:
            num_qubits: Number of qubits (must match input feature dimension).
            rotation_gate: Single-qubit rotation ('rx', 'ry', or 'rz').
            parameter_prefix: Prefix for input feature ParameterVector.
        """
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        rot = rotation_gate.lower()
        if rot not in self.SUPPORTED_ROTATIONS:
            raise ValueError(
                f"Unsupported rotation_gate '{rotation_gate}'. Supported: {self.SUPPORTED_ROTATIONS}"
            )

        self.num_qubits = num_qubits
        self.rotation_gate = rot
        self.parameter_prefix = parameter_prefix

        # Parameter vector representing input features
        self.parameters = ParameterVector(parameter_prefix, num_qubits)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Build the parameterized QuantumCircuit."""
        qc = QuantumCircuit(self.num_qubits, name=f"AngleEncoding_{self.rotation_gate.upper()}")
        for i in range(self.num_qubits):
            gate_func = getattr(qc, self.rotation_gate)
            gate_func(self.parameters[i], i)
        return qc

    def assign_features(self, feature_vector: np.ndarray | Sequence[float]) -> QuantumCircuit:
        """Bind a concrete feature vector to the parameterized encoding circuit.

        Args:
            feature_vector: 1D array of length num_qubits.

        Returns:
            Bound QuantumCircuit with numeric rotation angles.
        """
        features = np.asarray(feature_vector, dtype=float)
        if features.ndim != 1 or len(features) != self.num_qubits:
            raise ValueError(
                f"Expected 1D feature vector of length {self.num_qubits}, got shape {features.shape}"
            )
        param_dict = {self.parameters[i]: float(features[i]) for i in range(self.num_qubits)}
        return self.circuit.assign_parameters(param_dict)

    @property
    def num_parameters(self) -> int:
        """Number of input feature parameters (equals num_qubits)."""
        return self.num_qubits

    def __repr__(self) -> str:
        return f"AngleFeatureMap(num_qubits={self.num_qubits}, rotation='{self.rotation_gate}')"
