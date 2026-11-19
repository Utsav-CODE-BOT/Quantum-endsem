"""Fixed canonical Quantum Neural Network baselines for comparative benchmarking."""

from __future__ import annotations

from typing import Any
import numpy as np

from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.vqc_model import ConfigurableVQC
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_fixed_shallow_vqc(
    num_qubits: int = 4,
    maxiter: int = 35,
    random_seed: int = 42,
) -> ConfigurableVQC:
    """Fixed shallow VQC baseline: 1 layer, single RY rotation, linear entanglement."""
    model = ConfigurableVQC(
        num_qubits=num_qubits,
        num_layers=1,
        rotation_gates=["ry"],
        entanglement_type="linear",
        maxiter=maxiter,
        random_seed=random_seed,
    )
    setattr(model, "arch_id", "fixed_shallow_vqc_l1")
    return model


def create_fixed_standard_vqc(
    num_qubits: int = 4,
    maxiter: int = 35,
    random_seed: int = 42,
) -> ConfigurableVQC:
    """Fixed standard VQC baseline: 2 layers, RY-RZ rotations, linear entanglement."""
    model = ConfigurableVQC(
        num_qubits=num_qubits,
        num_layers=2,
        rotation_gates=["ry", "rz"],
        entanglement_type="linear",
        maxiter=maxiter,
        random_seed=random_seed,
    )
    setattr(model, "arch_id", "fixed_standard_vqc_l2")
    return model


def create_fixed_deep_vqc(
    num_qubits: int = 4,
    maxiter: int = 35,
    random_seed: int = 42,
) -> ConfigurableVQC:
    """Fixed deep VQC baseline: 4 layers, RY-RZ rotations, circular entanglement."""
    model = ConfigurableVQC(
        num_qubits=num_qubits,
        num_layers=4,
        rotation_gates=["ry", "rz"],
        entanglement_type="circular",
        maxiter=maxiter,
        random_seed=random_seed,
    )
    setattr(model, "arch_id", "fixed_deep_vqc_l4")
    return model
