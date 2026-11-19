"""Quantum execution backend manager for ideal and noisy simulations.

Provides standardized primitives:
- Statevector ideal Sampler (exact quasi-probabilities, deterministic).
- AerSimulator for shot-based and noisy circuit executions.
"""

from __future__ import annotations

from typing import Optional

from qiskit.primitives import BaseSampler, Sampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler as AerSampler

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class QuantumSimulatorManager:
    """Manages quantum simulator backends for ideal and noise-aware evaluations."""

    @staticmethod
    def get_ideal_sampler() -> BaseSampler:
        """Return the ideal statevector Sampler primitive for fast, noise-free evaluation.

        Returns:
            qiskit.primitives.Sampler instance.
        """
        logger.debug("Instantiating ideal statevector Sampler.")
        return Sampler()

    @staticmethod
    def get_aer_sampler(
        shots: int = 1024,
        noise_model: NoiseModel | None = None,
        seed_simulator: int | None = 42,
    ) -> BaseSampler:
        """Return an AerSampler supporting finite shot sampling and quantum noise models.

        Args:
            shots: Number of measurement shots.
            noise_model: Optional Qiskit Aer NoiseModel.
            seed_simulator: Seed for Aer simulation reproducibility.

        Returns:
            qiskit_aer.primitives.Sampler configured with backend options.
        """
        backend_options = {
            "method": "statevector" if noise_model is None else "density_matrix",
            "shots": shots,
        }
        if seed_simulator is not None:
            backend_options["seed_simulator"] = seed_simulator

        logger.debug(
            "Instantiating AerSampler with shots=%d, noise_model=%s",
            shots,
            "enabled" if noise_model is not None else "none",
        )
        aer_sampler = AerSampler(
            backend_options=backend_options,
            run_options={"shots": shots},
        )
        if noise_model is not None:
            aer_sampler.set_options(noise_model=noise_model)
        return aer_sampler
