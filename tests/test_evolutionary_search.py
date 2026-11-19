"""Unit and integration tests for Evolutionary Search components."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.classification_metrics import ClassificationReport
from src.evaluation.evaluator import ArchitectureEvaluator, CandidateEvaluationResult
from src.optimization.architecture_space import ArchitectureSpace
from src.optimization.crossover import ArchitectureCrossover
from src.optimization.evolutionary_search import EvolutionarySearchEngine
from src.optimization.mutation import ArchitectureMutation
from src.optimization.selection import TournamentSelection
from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.circuit_generator import CircuitMetadata
from src.utils.seed_manager import SeedManager


class TestGeneticOperators:
    """Test individual evolutionary operations."""

    @pytest.fixture
    def dummy_candidates(self) -> list[CandidateEvaluationResult]:
        meta = CircuitMetadata(4, 10, 20, 16, 4, {}, 4, 16)
        report = ClassificationReport(0.8, 0.8, 0.8, 0.8, [[5, 0], [0, 5]])
        candidates = []
        for i, fit in enumerate([0.2, 0.9, 0.5, 0.7]):
            cfg = ArchitectureConfig(4, 2, ("ry",), "linear")
            c = CandidateEvaluationResult(
                arch_id=f"c_{i}",
                config=cfg,
                circuit_metadata=meta,
                val_metrics=report,
                fitness=fit,
            )
            candidates.append(c)
        return candidates

    def test_tournament_selection(self, dummy_candidates) -> None:
        selector = TournamentSelection(tournament_size=4)
        rng = np.random.default_rng(42)
        # With tournament_size=4, winner must be the highest fitness candidate (fit=0.9, idx 1)
        winner = selector.select_parent(dummy_candidates, rng)
        assert winner.fitness == 0.9
        assert winner.arch_id == "c_1"

    def test_crossover_operation(self) -> None:
        p1 = ArchitectureConfig(4, 1, ("ry",), "linear", 1)
        p2 = ArchitectureConfig(4, 4, ("rx", "rz"), "circular", 2)

        cross = ArchitectureCrossover(crossover_rate=1.0)
        rng = np.random.default_rng(42)
        child1, child2 = cross.crossover(p1, p2, rng)

        assert isinstance(child1, ArchitectureConfig)
        assert isinstance(child2, ArchitectureConfig)
        # Child layers must be drawn from parents
        assert child1.num_layers in [1, 4]
        assert child2.num_layers in [1, 4]

    def test_mutation_operation(self) -> None:
        p = ArchitectureConfig(4, 2, ("ry",), "linear", 1)
        mut = ArchitectureMutation(mutation_rate=1.0)
        rng = np.random.default_rng(42)
        mutated = mut.mutate(p, rng)

        assert isinstance(mutated, ArchitectureConfig)
        assert mutated.num_qubits == 4
        assert 1 <= mutated.num_layers <= 5


class TestEvolutionarySearchIntegration:
    """End-to-end integration test of the evolutionary optimization loop."""

    def test_mini_evolutionary_search(self) -> None:
        X_train = np.array([
            [0.1, 0.2, 0.1, 0.2],
            [0.2, 0.1, 0.2, 0.1],
            [2.8, 2.9, 3.0, 2.8],
            [2.9, 3.0, 2.9, 3.0],
        ])
        y_train = np.array([0, 0, 1, 1])
        X_val = np.array([
            [0.15, 0.25, 0.1, 0.2],
            [3.0, 2.8, 2.8, 2.9],
        ])
        y_val = np.array([0, 1])

        evaluator = ArchitectureEvaluator(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            maxiter=5,
            enable_cache=True,
        )

        engine = EvolutionarySearchEngine(
            evaluator=evaluator,
            population_size=3,
            max_generations=2,
            elite_count=1,
            seed_manager=SeedManager(global_seed=42),
        )

        result = engine.run()

        assert result.best_candidate is not None
        assert len(result.generation_history) == 2
        assert result.total_evaluations > 0
        assert result.best_candidate.fitness > -1.0
        # Check elitism: generation 2 contains an elite candidate
        gen1_best = result.generation_history[0].best_candidate
        gen2_candidates = result.generation_history[1].candidates
        gen2_ids = [c.arch_id for c in gen2_candidates]
        assert gen1_best.arch_id in gen2_ids
