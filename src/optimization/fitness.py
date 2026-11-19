"""Normalized multi-objective fitness formulation and sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.quantum.circuit_generator import CircuitMetadata
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FitnessBreakdown:
    """Detailed components of the scalar fitness calculation."""

    scalar_fitness: float
    performance_score: float  # [0, 1]
    complexity_score: float  # [0, 1]
    robustness_score: float  # [0, 1]
    w_performance: float
    w_complexity: float
    w_robustness: float

    def to_dict(self) -> dict[str, float]:
        return {
            "scalar_fitness": self.scalar_fitness,
            "performance_score": self.performance_score,
            "complexity_score": self.complexity_score,
            "robustness_score": self.robustness_score,
            "w_performance": self.w_performance,
            "w_complexity": self.w_complexity,
            "w_robustness": self.w_robustness,
        }


class FitnessCalculator:
    """Calculates normalized resource-aware multi-objective fitness for VQC architectures.

    Mathematical formulation:
        Fitness(A) = w1 * Performance(A) - w2 * Complexity(A) + w3 * Robustness(A)

    All terms are strictly normalized to [0.0, 1.0]:
        - Performance: Validation accuracy in [0, 1]
        - Complexity: Normalized linear combination of circuit depth and 2-qubit (CX) gate count:
              Complexity = 0.5 * (depth / max_depth) + 0.5 * (cx_count / max_cx)
        - Robustness: Relative retention under simulated noise:
              Robustness = 1.0 - max(0.0, (perf_ideal - perf_noisy) / max(perf_ideal, 1e-6))
    """

    def __init__(
        self,
        w_performance: float = 0.5,
        w_complexity: float = 0.3,
        w_robustness: float = 0.2,
        max_depth: int = 50,
        max_cx_gates: int = 30,
    ) -> None:
        """Initialize calculator.

        Args:
            w_performance: Weight for classification accuracy (default: 0.5).
            w_complexity: Penalty weight for circuit complexity (default: 0.3).
            w_robustness: Weight for noise robustness (default: 0.2).
            max_depth: Upper bound for circuit depth normalization.
            max_cx_gates: Upper bound for two-qubit gate count normalization.
        """
        self.w_perf = float(w_performance)
        self.w_comp = float(w_complexity)
        self.w_rob = float(w_robustness)
        self.max_depth = max(max_depth, 1)
        self.max_cx = max(max_cx_gates, 1)

    def calculate_complexity(self, metadata: CircuitMetadata) -> float:
        """Compute normalized circuit complexity score in [0.0, 1.0]."""
        norm_depth = min(metadata.circuit_depth / self.max_depth, 1.0)
        norm_cx = min(metadata.two_qubit_gates / self.max_cx, 1.0)
        # Equal weighting between depth and 2-qubit gate overhead
        return float(0.5 * norm_depth + 0.5 * norm_cx)

    def calculate_robustness(
        self,
        ideal_accuracy: float,
        noisy_accuracy: float | None = None,
    ) -> float:
        """Compute normalized noise robustness score in [0.0, 1.0]."""
        if noisy_accuracy is None:
            return 1.0  # Default neutral robustness when noise evaluation is inactive
        drop = max(0.0, ideal_accuracy - noisy_accuracy)
        relative_drop = drop / max(ideal_accuracy, 1e-4)
        return float(max(0.0, min(1.0, 1.0 - relative_drop)))

    def evaluate(
        self,
        val_accuracy: float,
        metadata: CircuitMetadata,
        noisy_accuracy: float | None = None,
    ) -> FitnessBreakdown:
        """Compute complete scalar fitness and breakdown.

        Args:
            val_accuracy: Validation accuracy in [0.0, 1.0].
            metadata: CircuitMetadata for the candidate.
            noisy_accuracy: Optional noisy simulation accuracy.

        Returns:
            FitnessBreakdown containing scalar fitness and individual objective terms.
        """
        perf = float(max(0.0, min(1.0, val_accuracy)))
        comp = self.calculate_complexity(metadata)
        rob = self.calculate_robustness(perf, noisy_accuracy)

        # Multi-objective scalar fitness
        # Note: Added base offset + w_comp so fitness remains >= 0 for reasonable candidates
        scalar = (self.w_perf * perf) - (self.w_comp * comp) + (self.w_rob * rob)

        return FitnessBreakdown(
            scalar_fitness=float(scalar),
            performance_score=perf,
            complexity_score=comp,
            robustness_score=rob,
            w_performance=self.w_perf,
            w_complexity=self.w_comp,
            w_robustness=self.w_rob,
        )

    def sensitivity_analysis(
        self,
        val_accuracy: float,
        metadata: CircuitMetadata,
        weight_grid: list[tuple[float, float, float]],
        noisy_accuracy: float | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate how candidate fitness changes under varying objective weights."""
        results = []
        for w1, w2, w3 in weight_grid:
            temp_calc = FitnessCalculator(
                w_performance=w1,
                w_complexity=w2,
                w_robustness=w3,
                max_depth=self.max_depth,
                max_cx_gates=self.max_cx,
            )
            breakdown = temp_calc.evaluate(val_accuracy, metadata, noisy_accuracy)
            results.append({
                "weights": (w1, w2, w3),
                "scalar_fitness": breakdown.scalar_fitness,
                "perf_term": w1 * breakdown.performance_score,
                "comp_term": -w2 * breakdown.complexity_score,
                "rob_term": w3 * breakdown.robustness_score,
            })
        return results
