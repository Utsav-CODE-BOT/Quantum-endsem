"""Threaded evolutionary architecture search manager with live state reporting."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np

from src.evaluation.evaluator import ArchitectureEvaluator, CandidateEvaluationResult
from src.evaluation.fitness import FitnessCalculator
from src.optimization.architecture_space import ArchitectureSpace
from src.optimization.evolutionary_search import (
    EvolutionarySearchEngine,
    EvolutionarySearchResult,
    GenerationRecord,
)
from src.utils.config_loader import get_project_root
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SearchJobState:
    """Thread-safe state container for active or completed evolutionary search runs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status: str = "idle"  # "idle" | "running" | "completed" | "error" | "cancelled"
        self.current_generation: int = 0
        self.total_generations: int = 0
        self.current_candidate: int = 0
        self.total_candidates: int = 0
        self.best_fitness: float = 0.0
        self.best_val_accuracy: float = 0.0
        self.elapsed_time_seconds: float = 0.0
        self.generation_records: list[dict[str, Any]] = []
        self.evaluated_candidates: list[dict[str, Any]] = []
        self.final_result: dict[str, Any] | None = None
        self.error_message: str = ""
        self._cancel_requested: bool = False

    def update_generation(self, record: GenerationRecord, total_gens: int) -> None:
        with self.lock:
            self.current_generation = record.generation
            self.total_generations = total_gens
            self.best_fitness = float(record.best_fitness)
            self.best_val_accuracy = float(record.best_candidate.val_metrics.accuracy)
            self.generation_records.append(record.to_dict())
            for c in record.candidates:
                self.evaluated_candidates.append(c.to_dict())

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "current_generation": self.current_generation,
                "total_generations": self.total_generations,
                "best_fitness": round(self.best_fitness, 4),
                "best_val_accuracy": round(self.best_val_accuracy, 4),
                "elapsed_time_seconds": round(self.elapsed_time_seconds, 2),
                "num_evaluated_candidates": len(self.evaluated_candidates),
                "generation_records": list(self.generation_records),
                "evaluated_candidates": list(self.evaluated_candidates),
                "final_result": self.final_result,
                "error_message": self.error_message,
            }


class SearchService:
    """Service orchestrating background evolutionary searches and saved history retrieval."""

    _active_job: SearchJobState = SearchJobState()
    _worker_thread: threading.Thread | None = None

    @classmethod
    def get_status(cls) -> dict[str, Any]:
        """Return current status of the search engine."""
        return cls._active_job.to_dict()

    @classmethod
    def is_running(cls) -> bool:
        """Return whether an evolutionary search is currently executing."""
        with cls._active_job.lock:
            return cls._active_job.status == "running"

    @classmethod
    def cancel_search(cls) -> None:
        """Request cooperative cancellation of the running search."""
        with cls._active_job.lock:
            if cls._active_job.status == "running":
                cls._active_job._cancel_requested = True
                cls._active_job.status = "cancelled"
                logger.info("Evolutionary search cancellation requested by user.")

    @classmethod
    def load_saved_history(cls) -> dict[str, Any]:
        """Load historically saved evolutionary search results from disk."""
        path = get_project_root() / "experiments" / "results" / "search_history.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_saved_best_architecture(cls) -> dict[str, Any]:
        """Load historically discovered best architecture from disk."""
        path = get_project_root() / "experiments" / "results" / "best_architecture.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def start_search(
        cls,
        pop_size: int = 4,
        generations: int = 2,
        maxiter: int = 25,
        w_perf: float = 0.50,
        w_comp: float = 0.30,
        w_rob: float = 0.20,
        crossover_rate: float = 0.70,
        mutation_rate: float = 0.30,
        tournament_size: int = 2,
        random_seed: int = 42,
    ) -> bool:
        """Launch background evolutionary architecture search."""
        with cls._active_job.lock:
            if cls._active_job.status == "running":
                return False  # Already running

            cls._active_job = SearchJobState()
            cls._active_job.status = "running"
            cls._active_job.total_generations = generations

        def _worker():
            t0 = time.perf_counter()
            try:
                # 1. Load data
                data_path = get_project_root() / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
                data = np.load(data_path)
                X_train, y_train = data["X_train"], data["y_train"]
                X_val, y_val = data["X_val"], data["y_val"]

                # 2. Setup Evaluator
                fitness_calc = FitnessCalculator(
                    w_performance=w_perf,
                    w_complexity=w_comp,
                    w_robustness=w_rob,
                )
                evaluator = ArchitectureEvaluator(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    fitness_calculator=fitness_calc,
                    maxiter=maxiter,
                    optimizer_name="cobyla",
                    training_seed=random_seed,
                    enable_cache=True,
                )

                # 3. Setup Engine
                engine = EvolutionarySearchEngine(
                    evaluator=evaluator,
                    population_size=pop_size,
                    max_generations=generations,
                    elite_count=1,
                    stagnation_limit=generations,
                    crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate,
                    tournament_size=tournament_size,
                )

                # 4. Execute search with cooperative monitoring
                current_configs = engine.generator.sample_population(pop_size)
                generation_history: list[GenerationRecord] = []
                overall_best: CandidateEvaluationResult | None = None

                for gen in range(generations):
                    if cls._active_job._cancel_requested:
                        break

                    gen_start = time.perf_counter()
                    evaluated_pop = []
                    for cfg in current_configs:
                        if cls._active_job._cancel_requested:
                            break
                        res = evaluator.evaluate(cfg)
                        evaluated_pop.append(res)

                    if cls._active_job._cancel_requested:
                        break

                    evaluated_pop.sort(key=lambda c: c.fitness, reverse=True)
                    gen_best = evaluated_pop[0]
                    avg_fitness = float(np.mean([c.fitness for c in evaluated_pop]))
                    gen_time = time.perf_counter() - gen_start

                    rec = GenerationRecord(
                        generation=gen + 1,
                        best_candidate=gen_best,
                        best_fitness=gen_best.fitness,
                        avg_fitness=avg_fitness,
                        candidates=evaluated_pop,
                        elapsed_time_seconds=round(gen_time, 2),
                    )
                    generation_history.append(rec)
                    cls._active_job.update_generation(rec, generations)

                    if overall_best is None or gen_best.fitness > overall_best.fitness:
                        overall_best = gen_best

                    if gen == generations - 1:
                        break

                    # Offspring generation
                    next_configs = [evaluated_pop[0].config]  # Elitism = 1
                    while len(next_configs) < pop_size:
                        p1 = engine.selector.select_parent(evaluated_pop, engine.rng)
                        p2 = engine.selector.select_parent(evaluated_pop, engine.rng)
                        c1, c2 = engine.crossover_op.crossover(p1.config, p2.config, engine.rng)
                        next_configs.append(engine.mutation_op.mutate(c1, engine.rng))
                        if len(next_configs) < pop_size:
                            next_configs.append(engine.mutation_op.mutate(c2, engine.rng))
                    current_configs = next_configs

                total_time = time.perf_counter() - t0
                with cls._active_job.lock:
                    if not cls._active_job._cancel_requested:
                        cls._active_job.status = "completed"
                        cls._active_job.elapsed_time_seconds = total_time
                        if overall_best:
                            cls._active_job.final_result = overall_best.to_dict()

            except Exception as e:
                logger.error("Error in evolutionary search worker: %s", e)
                with cls._active_job.lock:
                    cls._active_job.status = "error"
                    cls._active_job.error_message = str(e)

        cls._worker_thread = threading.Thread(target=_worker, daemon=True)
        cls._worker_thread.start()
        return True
