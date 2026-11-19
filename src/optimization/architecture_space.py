"""Defines the discrete search space and structural validation for VQC architectures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.quantum.architecture_schema import ArchitectureConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureValidationError(Exception):
    """Raised when an architecture configuration violates search space bounds or structural validity."""


@dataclass
class ArchitectureSpace:
    """Discrete architectural hyperparameter bounds for the search space."""

    qubit_choices: tuple[int, ...] = (4,)  # Must match preprocessed feature dimension
    layer_min: int = 1
    layer_max: int = 5
    rotation_gate_options: tuple[tuple[str, ...], ...] = (
        ("ry",),
        ("rz",),
        ("ry", "rz"),
        ("rx", "ry"),
        ("rx", "rz"),
        ("rx", "ry", "rz"),
    )
    entanglement_types: tuple[str, ...] = ("linear", "circular", "pairwise", "none")
    entanglement_freq_options: tuple[int, ...] = (1, 2)
    feature_rotation_options: tuple[str, ...] = ("ry",)
    measurement_strategies: tuple[str, ...] = ("parity",)

    def validate(self, config: ArchitectureConfig) -> bool:
        """Validate that an architecture config conforms to all structural rules.

        Args:
            config: ArchitectureConfig to inspect.

        Returns:
            True if valid.

        Raises:
            ArchitectureValidationError: If any rule is violated.
        """
        # 1. Qubit bounds
        if config.num_qubits not in self.qubit_choices:
            raise ArchitectureValidationError(
                f"num_qubits {config.num_qubits} not in allowed choices {self.qubit_choices}"
            )

        # 2. Layer bounds
        if not (self.layer_min <= config.num_layers <= self.layer_max):
            raise ArchitectureValidationError(
                f"num_layers {config.num_layers} outside [{self.layer_min}, {self.layer_max}]"
            )

        # 3. Rotation gates
        if len(config.rotation_gates) == 0:
            raise ArchitectureValidationError("rotation_gates cannot be empty")
        for rot in config.rotation_gates:
            if rot.lower() not in ("rx", "ry", "rz"):
                raise ArchitectureValidationError(f"Invalid rotation gate '{rot}'")

        # 4. Entanglement type
        if config.entanglement_type.lower() not in self.entanglement_types:
            raise ArchitectureValidationError(
                f"Invalid entanglement_type '{config.entanglement_type}'. Allowed: {self.entanglement_types}"
            )

        # 5. Entanglement frequency
        if config.entanglement_freq not in self.entanglement_freq_options and config.entanglement_freq <= 0:
            raise ArchitectureValidationError(
                f"Invalid entanglement_freq {config.entanglement_freq}"
            )

        # 6. Feature rotation
        if config.feature_rotation.lower() not in ("rx", "ry", "rz"):
            raise ArchitectureValidationError(
                f"Invalid feature_rotation '{config.feature_rotation}'"
            )

        # 7. Measurement strategy
        if config.measurement_strategy.lower() not in ("parity", "single_qubit"):
            raise ArchitectureValidationError(
                f"Invalid measurement_strategy '{config.measurement_strategy}'"
            )

        return True
