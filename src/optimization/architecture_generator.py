"""Quantum neural architecture generator for candidate VQC configurations."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.optimization.architecture_space import (
    ArchitectureSpace,
    ArchitectureValidationError,
)
from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.circuit_generator import CircuitMetadata, VQCCircuitGenerator
from src.utils.logging_config import get_logger
from src.utils.seed_manager import SeedManager

logger = get_logger(__name__)


class ArchitectureGenerator:
    """Generates valid candidate VQC architectures deterministically from the defined search space."""

    def __init__(
        self,
        space: ArchitectureSpace | None = None,
        seed_manager: SeedManager | None = None,
    ) -> None:
        """Initialize generator.

        Args:
            space: ArchitectureSpace defining search bounds.
            seed_manager: SeedManager for reproducible pseudorandom choices.
        """
        self.space = space or ArchitectureSpace()
        self.seed_manager = seed_manager or SeedManager(global_seed=42)
        self.rng = self.seed_manager.get_numpy_rng("arch_generator")

    def sample_candidate(self) -> ArchitectureConfig:
        """Sample a single structurally valid candidate configuration.

        Returns:
            ArchitectureConfig instance.
        """
        num_qubits = int(self.rng.choice(self.space.qubit_choices))
        num_layers = int(self.rng.integers(self.space.layer_min, self.space.layer_max + 1))
        rot_idx = int(self.rng.integers(0, len(self.space.rotation_gate_options)))
        rotation_gates = self.space.rotation_gate_options[rot_idx]
        ent_type = str(self.rng.choice(self.space.entanglement_types))
        ent_freq = int(self.rng.choice(self.space.entanglement_freq_options))
        feat_rot = str(self.rng.choice(self.space.feature_rotation_options))
        meas_strat = str(self.rng.choice(self.space.measurement_strategies))

        config = ArchitectureConfig(
            num_qubits=num_qubits,
            num_layers=num_layers,
            rotation_gates=rotation_gates,
            entanglement_type=ent_type,
            entanglement_freq=ent_freq,
            feature_rotation=feat_rot,
            measurement_strategy=meas_strat,
        )

        self.space.validate(config)
        return config

    def sample_population(self, size: int) -> list[ArchitectureConfig]:
        """Sample a population of unique valid candidate configurations.

        Args:
            size: Desired number of distinct architectures.

        Returns:
            List of ArchitectureConfig instances with unique arch_ids.
        """
        population: list[ArchitectureConfig] = []
        seen_ids: set[str] = set()

        max_attempts = size * 50
        attempts = 0

        while len(population) < size and attempts < max_attempts:
            attempts += 1
            candidate = self.sample_candidate()
            if candidate.arch_id not in seen_ids:
                seen_ids.add(candidate.arch_id)
                population.append(candidate)

        if len(population) < size:
            logger.warning(
                "Requested population size %d, but only found %d unique architectures.",
                size,
                len(population),
            )

        logger.info("Sampled population of %d unique candidate architectures.", len(population))
        return population

    @staticmethod
    def compile_circuit(config: ArchitectureConfig) -> tuple[VQCCircuitGenerator, CircuitMetadata]:
        """Instantiate the circuit generator and extract metrics for a candidate.

        Args:
            config: ArchitectureConfig to compile.

        Returns:
            Tuple of (VQCCircuitGenerator, CircuitMetadata).
        """
        gen = VQCCircuitGenerator.from_specs(
            num_qubits=config.num_qubits,
            num_layers=config.num_layers,
            rotation_gates=config.rotation_gates,
            entanglement_type=config.entanglement_type,
            entanglement_freq=config.entanglement_freq,
            feature_rotation=config.feature_rotation,
        )
        return gen, gen.metadata
