"""Genetic crossover operators for quantum architecture configurations."""

from __future__ import annotations

import numpy as np

from src.optimization.architecture_space import ArchitectureSpace
from src.quantum.architecture_schema import ArchitectureConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ArchitectureCrossover:
    """Recombines architectural genes of two parent VQCs to generate valid offspring."""

    def __init__(
        self,
        crossover_rate: float = 0.7,
        space: ArchitectureSpace | None = None,
    ) -> None:
        """Initialize crossover operator.

        Args:
            crossover_rate: Probability of performing crossover (vs copying parents).
            space: ArchitectureSpace for structural validation.
        """
        self.crossover_rate = float(crossover_rate)
        self.space = space or ArchitectureSpace()

    def crossover(
        self,
        parent1: ArchitectureConfig,
        parent2: ArchitectureConfig,
        rng: np.random.Generator,
    ) -> tuple[ArchitectureConfig, ArchitectureConfig]:
        """Produce two offspring from parent1 and parent2.

        Args:
            parent1: First parent architecture.
            parent2: Second parent architecture.
            rng: NumPy Generator for random choices.

        Returns:
            Tuple of two valid offspring ArchitectureConfig instances.
        """
        if rng.random() > self.crossover_rate:
            # No crossover: return copies
            return parent1, parent2

        # Uniform gene exchange
        # Gene 1: num_layers
        l1, l2 = (parent1.num_layers, parent2.num_layers) if rng.random() < 0.5 else (parent2.num_layers, parent1.num_layers)
        # Gene 2: rotation_gates
        r1, r2 = (parent1.rotation_gates, parent2.rotation_gates) if rng.random() < 0.5 else (parent2.rotation_gates, parent1.rotation_gates)
        # Gene 3: entanglement_type
        e1, e2 = (parent1.entanglement_type, parent2.entanglement_type) if rng.random() < 0.5 else (parent2.entanglement_type, parent1.entanglement_type)
        # Gene 4: entanglement_freq
        f1, f2 = (parent1.entanglement_freq, parent2.entanglement_freq) if rng.random() < 0.5 else (parent2.entanglement_freq, parent1.entanglement_freq)

        offspring1 = ArchitectureConfig(
            num_qubits=parent1.num_qubits,
            num_layers=l1,
            rotation_gates=r1,
            entanglement_type=e1,
            entanglement_freq=f1,
            feature_rotation=parent1.feature_rotation,
            measurement_strategy=parent1.measurement_strategy,
        )

        offspring2 = ArchitectureConfig(
            num_qubits=parent2.num_qubits,
            num_layers=l2,
            rotation_gates=r2,
            entanglement_type=e2,
            entanglement_freq=f2,
            feature_rotation=parent2.feature_rotation,
            measurement_strategy=parent2.measurement_strategy,
        )

        # Validate offspring; fallback to parents if invalid
        try:
            self.space.validate(offspring1)
        except Exception:
            offspring1 = parent1

        try:
            self.space.validate(offspring2)
        except Exception:
            offspring2 = parent2

        return offspring1, offspring2
