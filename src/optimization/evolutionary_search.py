"""Custom evolutionary neural architecture search engine for Variational Quantum Classifiers.

Custom Python implementation:
- Initial population sampling across discrete VQC space
- Multi-objective fitness evaluation (Accuracy, Circuit Depth, Gate Count, Noise)
- Tournament parent selection
- Uniform architectural gene crossover
- Discrete gene mutation with structural repair
- Strict elitism preserving top performers across generations
- Early stopping upon fitness stagnation
- Full generation-by-generation trajectory logging
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.evaluator import ArchitectureEvaluator, CandidateEvaluationResult
from src.optimization.architecture_generator import ArchitectureGenerator
from src.optimization.architecture_space import ArchitectureSpace
from src.optimization.crossover import ArchitectureCrossover
from src.optimization.mutation import ArchitectureMutation
from src.optimization.selection import TournamentSelection
from src.quantum.architecture_schema import ArchitectureConfig
from src.utils.experiment_id import generate_experiment_id
from src.utils.logging_config import get_logger
from src.utils.seed_manager import SeedManager

logger = get_logger(__name__)


@dataclass
class GenerationRecord:
    """Snapshot of a single evolutionary generation."""

    generation: int
    best_candidate: CandidateEvaluationResult
    best_fitness: float
    avg_fitness: float
    candidates: list[CandidateEvaluationResult]
    elapsed_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "best_arch_id": self.best_candidate.arch_id,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "elapsed_time_seconds": self.elapsed_time_seconds,
            "candidate_ids": [c.arch_id for c in self.candidates],
            "candidate_fitnesses": [c.fitness for c in self.candidates],
        }


@dataclass
class EvolutionarySearchResult:
    """Final result of the evolutionary architecture optimization run."""

    best_candidate: CandidateEvaluationResult
    generation_history: list[GenerationRecord]
    total_evaluations: int
    unique_architectures_evaluated: int
    cache_hits: int
    total_runtime_seconds: float
    converged: bool
    stop_reason: str
    experiment_run_id: str = field(default_factory=generate_experiment_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_run_id": self.experiment_run_id,
            "best_candidate": self.best_candidate.to_dict(),
            "generation_history": [g.to_dict() for g in self.generation_history],
            "total_evaluations": self.total_evaluations,
            "unique_architectures_evaluated": self.unique_architectures_evaluated,
            "cache_hits": self.cache_hits,
            "total_runtime_seconds": self.total_runtime_seconds,
            "converged": self.converged,
            "stop_reason": self.stop_reason,
        }

    def save(self, path: str | Path) -> None:
        """Save complete search history to JSON atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f".tmp.{os.getpid()}_{time.time_ns()}")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, path)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise
        logger.info("Saved search history atomically to '%s'", path)


class EvolutionarySearchEngine:
    """Custom Evolutionary Algorithm optimizing VQC architectures."""

    def __init__(
        self,
        evaluator: ArchitectureEvaluator,
        population_size: int = 10,
        max_generations: int = 5,
        elite_count: int = 2,
        stagnation_limit: int = 3,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
        tournament_size: int = 3,
        space: ArchitectureSpace | None = None,
        seed_manager: SeedManager | None = None,
        experiment_run_id: str | None = None,
    ) -> None:
        """Initialize evolutionary search engine.

        Args:
            evaluator: ArchitectureEvaluator for candidate training and validation.
            population_size: Number of candidate architectures per generation.
            max_generations: Maximum generations to run.
            elite_count: Top-performing candidates preserved unchanged.
            stagnation_limit: Stop if best fitness does not improve for N generations.
            crossover_rate: Probability of crossover.
            mutation_rate: Probability of mutation.
            tournament_size: Tournament selection size.
            space: ArchitectureSpace definition.
            seed_manager: SeedManager for reproducibility.
            experiment_run_id: Optional unique experiment execution ID. If None, generated.
        """
        self.experiment_run_id = experiment_run_id or generate_experiment_id()
        self.evaluator = evaluator
        self.pop_size = max(population_size, 2)
        self.max_generations = max_generations
        self.elite_count = min(elite_count, self.pop_size - 1)
        self.stagnation_limit = stagnation_limit
        self.space = space or ArchitectureSpace()
        self.seed_mgr = seed_manager or SeedManager(global_seed=42)
        self.rng = self.seed_mgr.get_numpy_rng("evolutionary_search")

        self.generator = ArchitectureGenerator(space=self.space, seed_manager=self.seed_mgr)
        self.selector = TournamentSelection(tournament_size=tournament_size)
        self.crossover_op = ArchitectureCrossover(crossover_rate=crossover_rate, space=self.space)
        self.mutation_op = ArchitectureMutation(mutation_rate=mutation_rate, space=self.space)

    def run(self) -> EvolutionarySearchResult:
        """Execute the complete evolutionary architecture search.

        Returns:
            EvolutionarySearchResult containing best candidate and full search trajectory.
        """
        t_start = time.perf_counter()
        logger.info(
            "Starting Evolutionary Architecture Search: Pop=%d, Gen=%d, Elites=%d",
            self.pop_size,
            self.max_generations,
            self.elite_count,
        )

        # 1. Initialize random population
        current_configs = self.generator.sample_population(self.pop_size)
        generation_history: list[GenerationRecord] = []

        overall_best: CandidateEvaluationResult | None = None
        generations_without_improvement = 0
        converged = False
        stop_reason = "max_generations_reached"

        for gen in range(self.max_generations):
            gen_start = time.perf_counter()
            logger.info("=== Generation %d / %d ===", gen + 1, self.max_generations)

            # 2. Evaluate current population
            evaluated_population: list[CandidateEvaluationResult] = []
            for i, config in enumerate(current_configs):
                res = self.evaluator.evaluate(config)
                evaluated_population.append(res)
                logger.debug(
                    "Gen %d, Ind %d [%s]: Fitness=%.4f (Acc=%.2f%%, Depth=%d, Gates=%d)",
                    gen + 1,
                    i + 1,
                    res.arch_id,
                    res.fitness,
                    res.val_metrics.accuracy * 100,
                    res.circuit_metadata.circuit_depth,
                    res.circuit_metadata.total_gates,
                )

            # Filter successful candidates with finite fitness for ranking and reproduction
            successful_pop = [
                c for c in evaluated_population
                if c.success and math.isfinite(c.fitness)
            ]
            if not successful_pop:
                logger.error("All candidate evaluations failed in generation %d.", gen + 1)
                if overall_best is None:
                    raise RuntimeError(f"All candidate evaluations failed in generation {gen + 1}; cannot continue search.")
                logger.warning("Generation %d had no successful candidates; terminating search with existing best.", gen + 1)
                break

            # Sort successful candidates descending by fitness
            successful_pop.sort(key=lambda c: c.fitness, reverse=True)
            gen_best = successful_pop[0]
            avg_fitness = float(np.mean([c.fitness for c in successful_pop]))
            gen_time = time.perf_counter() - gen_start

            gen_record = GenerationRecord(
                generation=gen + 1,
                best_candidate=gen_best,
                best_fitness=gen_best.fitness,
                avg_fitness=avg_fitness,
                candidates=evaluated_population,
                elapsed_time_seconds=round(gen_time, 2),
            )
            generation_history.append(gen_record)

            logger.info(
                "Gen %d Summary: Best Fitness=%.4f (Arch: %s, Val Acc=%.2f%%) | Avg=%.4f (Time=%.2fs)",
                gen + 1,
                gen_best.fitness,
                gen_best.arch_id,
                gen_best.val_metrics.accuracy * 100,
                avg_fitness,
                gen_time,
            )

            # Track global best and stagnation
            if overall_best is None or gen_best.fitness > overall_best.fitness:
                overall_best = gen_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.stagnation_limit and gen < self.max_generations - 1:
                logger.info(
                    "Stagnation limit reached (%d generations without improvement). Terminating early.",
                    self.stagnation_limit,
                )
                converged = True
                stop_reason = "stagnation_limit_reached"
                break

            # If last generation, skip reproduction
            if gen == self.max_generations - 1:
                break

            # 3. Elitism: preserve top elite candidates unchanged from successful candidates
            actual_elites = min(self.elite_count, len(successful_pop))
            next_configs: list[ArchitectureConfig] = [
                successful_pop[e].config for e in range(actual_elites)
            ]

            # 4. Selection, Crossover, and Mutation for offspring from valid parents only
            while len(next_configs) < self.pop_size:
                p1 = self.selector.select_parent(successful_pop, self.rng)
                p2 = self.selector.select_parent(successful_pop, self.rng)

                child1, child2 = self.crossover_op.crossover(p1.config, p2.config, self.rng)
                child1 = self.mutation_op.mutate(child1, self.rng)
                child2 = self.mutation_op.mutate(child2, self.rng)

                next_configs.append(child1)
                if len(next_configs) < self.pop_size:
                    next_configs.append(child2)

            current_configs = next_configs

        total_runtime = time.perf_counter() - t_start
        assert overall_best is not None

        result = EvolutionarySearchResult(
            best_candidate=overall_best,
            generation_history=generation_history,
            total_evaluations=self.evaluator.total_evaluations,
            unique_architectures_evaluated=len(self.evaluator.cache),
            cache_hits=self.evaluator.cache_hits,
            total_runtime_seconds=round(total_runtime, 2),
            converged=converged,
            stop_reason=stop_reason,
            experiment_run_id=self.experiment_run_id,
        )

        logger.info(
            "Search Completed: Overall Best %s | Fitness=%.4f (Val Acc=%.2f%%, Depth=%d) in %.2fs",
            overall_best.arch_id,
            overall_best.fitness,
            overall_best.val_metrics.accuracy * 100,
            overall_best.circuit_metadata.circuit_depth,
            total_runtime,
        )
        return result
