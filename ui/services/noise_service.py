"""Quantum hardware noise analysis service connecting to Qiskit Aer simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from src.evaluation.noise_evaluator import (
    NoiseEvaluationReport,
    NoiseEvaluator,
    NoiseModelBuilder,
    NoiseModelConfig,
)
from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.vqc_model import ConfigurableVQC
from src.utils.config_loader import get_project_root
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class NoiseService:
    """Service providing noise model construction, live simulation, and saved results access."""

    @staticmethod
    def get_canonical_tiers() -> dict[str, NoiseModelConfig]:
        """Return standardized NISQ noise tiers benchmarked in the research project."""
        return {
            "Low Noise": NoiseModelConfig(
                name="Low_Noise",
                single_qubit_error=0.0005,
                two_qubit_error=0.005,
                measurement_error=0.005,
                shots=1024,
            ),
            "Medium Noise": NoiseModelConfig(
                name="Medium_Noise",
                single_qubit_error=0.0010,
                two_qubit_error=0.010,
                measurement_error=0.015,
                shots=1024,
            ),
            "High Noise": NoiseModelConfig(
                name="High_Noise",
                single_qubit_error=0.0050,
                two_qubit_error=0.030,
                measurement_error=0.030,
                shots=1024,
            ),
        }

    @staticmethod
    def load_saved_results() -> dict[str, Any]:
        """Load historically evaluated noise results from disk."""
        path = get_project_root() / "experiments" / "results" / "noise_experiment_results.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def evaluate_architecture_noise(
        cls,
        arch_config: ArchitectureConfig,
        noise_config: NoiseModelConfig,
        maxiter: int = 25,
    ) -> NoiseEvaluationReport:
        """Fit a VQC model on the training set and evaluate paired ideal vs noisy performance.

        Args:
            arch_config: Target candidate quantum circuit architecture.
            noise_config: Selected noise configuration.
            maxiter: Training iterations.

        Returns:
            NoiseEvaluationReport containing ideal vs noisy accuracy and retention index.
        """
        # Load dataset
        data_path = get_project_root() / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
        data = np.load(data_path)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

        # 1. Instantiate and fit VQC model
        vqc = ConfigurableVQC(
            num_qubits=arch_config.num_qubits,
            num_layers=arch_config.num_layers,
            rotation_gates=arch_config.rotation_gates,
            entanglement_type=arch_config.entanglement_type,
            entanglement_freq=arch_config.entanglement_freq,
            feature_rotation=arch_config.feature_rotation,
            maxiter=maxiter,
            random_seed=42,
        )
        setattr(vqc, "arch_id", arch_config.arch_id)
        vqc.fit(X_train, y_train)

        # 2. Evaluate under noise
        evaluator = NoiseEvaluator(seed=42)
        return evaluator.evaluate(vqc, X_test, y_test, noise_config)
