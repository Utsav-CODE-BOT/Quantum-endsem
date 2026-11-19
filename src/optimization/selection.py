"""Tournament selection for evolutionary architecture search."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.evaluation.evaluator import CandidateEvaluationResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TournamentSelection:
    """Selects high-performing parent architectures via tournament selection."""

    def __init__(self, tournament_size: int = 3) -> None:
        """Initialize selector.

        Args:
            tournament_size: Number of candidates randomly sampled per tournament.
        """
        if tournament_size <= 0:
            raise ValueError(f"tournament_size must be positive, got {tournament_size}")
        self.tournament_size = tournament_size

    def select_parent(
        self,
        population: Sequence[CandidateEvaluationResult],
        rng: np.random.Generator,
    ) -> CandidateEvaluationResult:
        """Select a single parent candidate from the population.

        Args:
            population: Evaluated candidates with valid fitness values.
            rng: NumPy Generator for random choice.

        Returns:
            Winning CandidateEvaluationResult.
        """
        if not population:
            raise ValueError("Cannot select from an empty population.")

        k = min(self.tournament_size, len(population))
        tournament_indices = rng.choice(len(population), size=k, replace=False)
        tournament = [population[i] for i in tournament_indices]

        # Winner has the highest scalar fitness
        winner = max(tournament, key=lambda c: c.fitness)
        return winner

    def select_pairs(
        self,
        population: Sequence[CandidateEvaluationResult],
        num_pairs: int,
        rng: np.random.Generator,
    ) -> list[tuple[CandidateEvaluationResult, CandidateEvaluationResult]]:
        """Select parent pairs for crossover."""
        pairs = []
        for _ in range(num_pairs):
            p1 = self.select_parent(population, rng)
            p2 = self.select_parent(population, rng)
            pairs.append((p1, p2))
        return pairs
