"""Normalized multi-objective fitness formulation and sensitivity analysis."""

from __future__ import annotations

import math
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
        depth = metadata.circuit_depth
        cx = metadata.two_qubit_gates
        if not math.isfinite(depth) or not math.isfinite(cx):
            return 1.0
        depth_clamped = max(0.0, float(depth))
        cx_clamped = max(0.0, float(cx))
        norm_depth = min(depth_clamped / self.max_depth, 1.0)
        norm_cx = min(cx_clamped / self.max_cx, 1.0)
        return float(0.5 * norm_depth + 0.5 * norm_cx)

    def calculate_robustness(
        self,
        ideal_accuracy: float,
        noisy_accuracy: float | None = None,
    ) -> float:
        """Compute normalized noise robustness score in [0.0, 1.0].

        Defined as:
            R(A) = min(1.0, max(0.0, Acc_noisy(A) / Acc_val(A))) for Acc_val(A) > 0.
            R(A) = 0.0 when Acc_val(A) <= 0.0, noisy_accuracy is None, or values are non-finite.

        When ideal validation accuracy is zero or non-positive, the candidate has zero baseline
        classification capacity, so retention under noise is zero by definition.

        Args:
            ideal_accuracy: Validation accuracy without hardware noise.
            noisy_accuracy: Validation accuracy under simulated NISQ noise. If None (unmeasured
                or failed), returns 0.0 to strictly prevent rewarding missing measurements.
        """
        if noisy_accuracy is None:
            return 0.0
        if not math.isfinite(ideal_accuracy) or not math.isfinite(noisy_accuracy):
            return 0.0
        if ideal_accuracy <= 0.0 or noisy_accuracy <= 0.0:
            return 0.0

        ideal_clamped = min(1.0, float(ideal_accuracy))
        noisy_clamped = min(1.0, float(noisy_accuracy))
        ratio = noisy_clamped / ideal_clamped
        return float(min(1.0, max(0.0, ratio)))

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
        if not math.isfinite(val_accuracy):
            perf = 0.0
        else:
            perf = float(max(0.0, min(1.0, val_accuracy)))
        comp = self.calculate_complexity(metadata)
        rob = self.calculate_robustness(perf, noisy_accuracy)

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
