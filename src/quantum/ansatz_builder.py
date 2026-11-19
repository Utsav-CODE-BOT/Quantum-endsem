"""Configurable variational ansatz builder for Variational Quantum Classifiers.

Supports:
- Configurable qubit counts and layer depths.
- Flexible single-qubit rotation combinations (e.g., ['ry'], ['ry', 'rz'], ['rx', 'ry', 'rz']).
- Distinct entanglement topologies:
    - 'linear': CNOTs between adjacent qubits (0->1, 1->2, ...).
    - 'circular': Linear + wrap-around CNOT ((n-1)->0).
    - 'pairwise': Disjoint pairs ((0,1), (2,3), then (1,2), (3,4) in alternating layers).
    - 'none': No entangling gates (product state ansatz).
- Configurable entanglement frequency (entangle every k layers).
- Parameter prefix 'θ' isolating trainable weights from input features.
"""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AnsatzBuilder:
    """Builds parameterized variational quantum ansatz circuits."""

    SUPPORTED_ROTATIONS = ("rx", "ry", "rz")
    SUPPORTED_ENTANGLEMENT = ("linear", "circular", "pairwise", "none")

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 2,
        rotation_gates: Sequence[str] = ("ry", "rz"),
        entanglement_type: str = "linear",
        entanglement_freq: int = 1,
        parameter_prefix: str = "θ",
    ) -> None:
        """Initialize the ansatz builder.

        Args:
            num_qubits: Number of qubits.
            num_layers: Number of variational layers.
            rotation_gates: Sequence of rotation gates applied per layer per qubit.
            entanglement_type: 'linear', 'circular', 'pairwise', or 'none'.
            entanglement_freq: Entanglement applied every k layers (1 = every layer).
            parameter_prefix: Prefix for trainable parameter vector.
        """
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if entanglement_freq <= 0:
            raise ValueError(f"entanglement_freq must be positive, got {entanglement_freq}")

        cleaned_rotations = [r.lower() for r in rotation_gates]
        for r in cleaned_rotations:
            if r not in self.SUPPORTED_ROTATIONS:
                raise ValueError(
                    f"Unsupported rotation gate '{r}'. Supported: {self.SUPPORTED_ROTATIONS}"
                )

        ent_type = entanglement_type.lower()
        if ent_type not in self.SUPPORTED_ENTANGLEMENT:
            raise ValueError(
                f"Unsupported entanglement_type '{entanglement_type}'. Supported: {self.SUPPORTED_ENTANGLEMENT}"
            )

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.rotation_gates = list(cleaned_rotations)
        self.entanglement_type = ent_type
        self.entanglement_freq = entanglement_freq
        self.parameter_prefix = parameter_prefix

        # Trainable parameters calculation
        # Each layer applies len(rotation_gates) to each of the num_qubits
        # Plus an optional final rotation layer after the last entanglement
        self.params_per_layer = num_qubits * len(self.rotation_gates)
        # Total parameters = (num_layers + 1) * params_per_layer
        self.total_parameters = (num_layers + 1) * self.params_per_layer
        self.parameters = ParameterVector(parameter_prefix, self.total_parameters)

        self.circuit = self._build_ansatz()

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the variational QuantumCircuit."""
        qc = QuantumCircuit(
            self.num_qubits,
            name=f"Ansatz_{self.entanglement_type}_L{self.num_layers}",
        )
        param_idx = 0

        # Layer loop
        for layer in range(self.num_layers):
            # 1. Parameterized rotation gates
            for rot in self.rotation_gates:
                gate_func = getattr(qc, rot)
                for q in range(self.num_qubits):
                    gate_func(self.parameters[param_idx], q)
                    param_idx += 1

            # 2. Entanglement block (if layer aligns with entanglement_freq)
            if (layer + 1) % self.entanglement_freq == 0 and self.entanglement_type != "none":
                self._apply_entanglement(qc, layer)

        # Final rotation layer
        for rot in self.rotation_gates:
            gate_func = getattr(qc, rot)
            for q in range(self.num_qubits):
                gate_func(self.parameters[param_idx], q)
                param_idx += 1

        assert param_idx == self.total_parameters, (
            f"Parameter count mismatch: used {param_idx}, allocated {self.total_parameters}"
        )
        return qc

    def _apply_entanglement(self, qc: QuantumCircuit, layer_idx: int) -> None:
        """Apply the specified two-qubit entangling gates."""
        n = self.num_qubits
        if n < 2:
            return  # Cannot entangle single qubit

        if self.entanglement_type == "linear":
            for q in range(n - 1):
                qc.cx(q, q + 1)

        elif self.entanglement_type == "circular":
            for q in range(n - 1):
                qc.cx(q, q + 1)
            qc.cx(n - 1, 0)

        elif self.entanglement_type == "pairwise":
            # Alternating even/odd pairs
            if layer_idx % 2 == 0:
                # Even pairs: (0, 1), (2, 3), ...
                for q in range(0, n - 1, 2):
                    qc.cx(q, q + 1)
            else:
                # Odd pairs: (1, 2), (3, 4), ...
                for q in range(1, n - 1, 2):
                    qc.cx(q, q + 1)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable variational parameters."""
        return self.total_parameters

    def __repr__(self) -> str:
        return (
            f"AnsatzBuilder(qubits={self.num_qubits}, layers={self.num_layers}, "
            f"rotations={self.rotation_gates}, entanglement='{self.entanglement_type}', "
            f"params={self.total_parameters})"
        )
