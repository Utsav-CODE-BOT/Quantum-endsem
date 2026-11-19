"""Genetic mutation operators for quantum architecture configurations."""

from __future__ import annotations

import numpy as np

from src.optimization.architecture_space import ArchitectureSpace
from src.quantum.architecture_schema import ArchitectureConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureMutation:
    """Mutates architectural genes to maintain diversity in the candidate population."""

    def __init__(
        self,
        mutation_rate: float = 0.3,
        space: ArchitectureSpace | None = None,
    ) -> None:
        """Initialize mutation operator.

        Args:
            mutation_rate: Probability of mutating a candidate.
            space: ArchitectureSpace for discrete gene options and validation.
        """
        self.mutation_rate = float(mutation_rate)
        self.space = space or ArchitectureSpace()

    def mutate(
        self,
        config: ArchitectureConfig,
        rng: np.random.Generator,
    ) -> ArchitectureConfig:
        """Apply random mutations to architectural genes.

        Args:
            config: ArchitectureConfig candidate.
            rng: NumPy Generator for random choices.

        Returns:
            Mutated, structurally valid ArchitectureConfig.
        """
        if rng.random() > self.mutation_rate:
            return config  # No mutation

        num_layers = config.num_layers
        rotation_gates = config.rotation_gates
        ent_type = config.entanglement_type
        ent_freq = config.entanglement_freq

        # Gene 1: Mutate depth (prob 0.4)
        if rng.random() < 0.4:
            delta = int(rng.choice([-1, 1]))
            new_layers = num_layers + delta
            if self.space.layer_min <= new_layers <= self.space.layer_max:
                num_layers = new_layers

        # Gene 2: Mutate rotation gates (prob 0.3)
        if rng.random() < 0.3:
            available = [r for r in self.space.rotation_gate_options if r != rotation_gates]
            if available:
                idx = int(rng.integers(0, len(available)))
                rotation_gates = available[idx]

        # Gene 3: Mutate entanglement topology (prob 0.3)
        if rng.random() < 0.3:
            available_ent = [e for e in self.space.entanglement_types if e != ent_type]
            if available_ent:
                ent_type = str(rng.choice(available_ent))

        # Gene 4: Mutate entanglement frequency (prob 0.2)
        if rng.random() < 0.2:
            available_freq = [f for f in self.space.entanglement_freq_options if f != ent_freq]
            if available_freq:
                ent_freq = int(rng.choice(available_freq))

        mutated = ArchitectureConfig(
            num_qubits=config.num_qubits,
            num_layers=num_layers,
            rotation_gates=rotation_gates,
            entanglement_type=ent_type,
            entanglement_freq=ent_freq,
            feature_rotation=config.feature_rotation,
            measurement_strategy=config.measurement_strategy,
        )

        try:
            self.space.validate(mutated)
            return mutated
        except Exception:
            return config  # Fallback to original if mutated is invalid
