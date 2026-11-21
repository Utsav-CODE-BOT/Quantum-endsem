"""Threaded evolutionary architecture search manager with live state reporting."""

from __future__ import annotations

import json
import os
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
from src.utils.experiment_id import generate_experiment_id
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
        self.experiment_run_id: str = ""
        self.persistence_status: str = "idle"  # "idle" | "persisting" | "persisted" | "failed" | "not_persisted"
        self.persistence_error: str = ""

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
                "experiment_run_id": self.experiment_run_id,
                "persistence_status": self.persistence_status,
                "persistence_error": self.persistence_error,
            }


class SearchService:
    """Service orchestrating background evolutionary searches and saved history retrieval."""

    _lock: threading.Lock = threading.Lock()
    _persistence_lock: threading.Lock = threading.Lock()
    _active_job: SearchJobState = SearchJobState()
    _worker_thread: threading.Thread | None = None

    @classmethod
    def get_status(cls) -> dict[str, Any]:
        """Return current status of the search engine."""
        return cls._active_job.to_dict()

    @classmethod
    def is_running(cls) -> bool:
        """Return whether an evolutionary search is currently executing."""
        with cls._lock:
            if cls._worker_thread is not None and cls._worker_thread.is_alive():
                return True
            with cls._active_job.lock:
                return cls._active_job.status in ("running", "cancelling")

    @classmethod
    def cancel_search(cls) -> None:
        """Request cooperative cancellation of the running search."""
        with cls._lock:
            with cls._active_job.lock:
                if cls._active_job.status == "running":
                    cls._active_job._cancel_requested = True
                    cls._active_job.status = "cancelling"
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
    def persist_search_artifacts(
        cls,
        search_result: EvolutionarySearchResult,
        overall_best: CandidateEvaluationResult,
        current_job: SearchJobState,
        hist_path: Path | None = None,
        best_arch_path: Path | None = None,
    ) -> bool:
        """Persist search history trajectory and conditionally update best architecture on disk atomically."""
        with cls._persistence_lock:
            target_hist = hist_path or (get_project_root() / "experiments" / "results" / "search_history.json")
            target_best = best_arch_path or (get_project_root() / "experiments" / "results" / "best_architecture.json")
            target_hist.parent.mkdir(parents=True, exist_ok=True)
            target_best.parent.mkdir(parents=True, exist_ok=True)

            # Atomic save of search history
            search_result.save(target_hist)

            should_update = True
            if target_best.exists():
                try:
                    with open(target_best, "r", encoding="utf-8") as f:
                        curr = json.load(f)

                    # Required Correction C — Verify evaluation condition compatibility
                    curr_weights = None
                    if "fitness_breakdown" in curr and isinstance(curr["fitness_breakdown"], dict):
                        curr_weights = (
                            curr["fitness_breakdown"].get("w_performance"),
                            curr["fitness_breakdown"].get("w_complexity"),
                            curr["fitness_breakdown"].get("w_robustness"),
                        )
                    new_weights = None
                    if overall_best.fitness_breakdown is not None:
                        new_weights = (
                            overall_best.fitness_breakdown.w_performance,
                            overall_best.fitness_breakdown.w_complexity,
                            overall_best.fitness_breakdown.w_robustness,
                        )

                    # If both provide fitness weights and they differ, they are not comparable
                    if curr_weights is not None and new_weights is not None and curr_weights != new_weights:
                        should_update = False
                        logger.info(
                            "Existing best architecture was evaluated under different objective weights (%s vs %s); skipping overwrite to preserve historical comparability.",
                            curr_weights,
                            new_weights,
                        )
                    elif curr.get("fitness", -1.0) > overall_best.fitness:
                        should_update = False
                        logger.info(
                            "Existing best architecture on disk has higher fitness (%.4f > %.4f); not overwriting.",
                            curr.get("fitness", 0),
                            overall_best.fitness,
                        )
                except Exception as e:
                    logger.warning("Could not read or parse existing best architecture for comparison: %s", e)

            if should_update:
                arch_dict = overall_best.to_dict()
                arch_dict["experiment_run_id"] = current_job.experiment_run_id
                temp_best = target_best.with_suffix(f".tmp.{os.getpid()}_{time.time_ns()}")
                try:
                    with open(temp_best, "w", encoding="utf-8") as f:
                        json.dump(arch_dict, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_best, target_best)
                except Exception:
                    if temp_best.exists():
                        try:
                            temp_best.unlink()
                        except OSError:
                            pass
                    raise

            return should_update

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
        experiment_run_id: str | None = None,
    ) -> bool:
        """Launch background evolutionary architecture search with dedicated job state."""
        with cls._lock:
            if cls._worker_thread is not None and cls._worker_thread.is_alive():
                logger.warning("Attempted to start search while worker thread is still active.")
                return False

            with cls._active_job.lock:
                if cls._active_job.status in ("running", "cancelling"):
                    return False

            # Create an independent state container for this specific search run
            run_id = experiment_run_id or generate_experiment_id()
            job = SearchJobState()
            job.status = "running"
            job.total_generations = generations
            job.experiment_run_id = run_id
            cls._active_job = job

        cls._worker_thread = threading.Thread(
            target=cls._run_worker,
            args=(
                job,
                pop_size,
                generations,
                maxiter,
                w_perf,
                w_comp,
                w_rob,
                crossover_rate,
                mutation_rate,
                tournament_size,
                random_seed,
            ),
            daemon=True,
        )
        cls._worker_thread.start()
        return True

    @classmethod
    def _run_worker(
        cls,
        current_job: SearchJobState,
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
    ) -> None:
        t0 = time.perf_counter()
        try:
            # 1. Load data
            data_path = get_project_root() / "data" / "processed" / "fashion_mnist_quantum_ready.npz"
            data = np.load(data_path)
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]

            # 2. Setup Evaluator with active noise evaluation
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
                evaluate_noise=True,
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
                if current_job._cancel_requested:
                    break

                gen_start = time.perf_counter()
                evaluated_pop = []
                for cfg in current_configs:
                    if current_job._cancel_requested:
                        break
                    res = evaluator.evaluate(cfg)
                    evaluated_pop.append(res)

                if current_job._cancel_requested:
                    break

                # Exclude failed candidates from ranking and reproduction
                successful_pop = [c for c in evaluated_pop if c.success]
                if not successful_pop:
                    err_msg = f"All candidate evaluations failed in generation {gen + 1}."
                    logger.error(err_msg)
                    with current_job.lock:
                        current_job.status = "error"
                        current_job.persistence_status = "not_persisted"
                        current_job.error_message = err_msg
                    return

                successful_pop.sort(key=lambda c: c.fitness, reverse=True)
                gen_best = successful_pop[0]
                avg_fitness = float(np.mean([c.fitness for c in successful_pop]))
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
                current_job.update_generation(rec, generations)

                if overall_best is None or gen_best.fitness > overall_best.fitness:
                    overall_best = gen_best

                if gen == generations - 1:
                    break

                # Offspring generation from valid parents only
                next_configs = [successful_pop[0].config]  # Elitism = 1 from valid candidates
                while len(next_configs) < pop_size:
                    p1 = engine.selector.select_parent(successful_pop, engine.rng)
                    p2 = engine.selector.select_parent(successful_pop, engine.rng)
                    c1, c2 = engine.crossover_op.crossover(p1.config, p2.config, engine.rng)
                    next_configs.append(engine.mutation_op.mutate(c1, engine.rng))
                    if len(next_configs) < pop_size:
                        next_configs.append(engine.mutation_op.mutate(c2, engine.rng))
                current_configs = next_configs

            total_time = time.perf_counter() - t0
            with current_job.lock:
                if current_job._cancel_requested:
                    current_job.status = "cancelled"
                    current_job.persistence_status = "not_persisted"
                    current_job.elapsed_time_seconds = total_time
                    logger.info("Evolutionary search was cancelled gracefully.")
                    return

                if overall_best is None or not overall_best.success:
                    current_job.status = "error"
                    current_job.persistence_status = "not_persisted"
                    current_job.error_message = "No valid architecture candidate succeeded during search."
                    logger.error(current_job.error_message)
                    return

                current_job.status = "completed"
                current_job.elapsed_time_seconds = total_time
                current_job.final_result = overall_best.to_dict()

            # Persist UI search results if successful and not cancelled
            if overall_best is not None and overall_best.success and generation_history:
                with current_job.lock:
                    current_job.persistence_status = "persisting"
                try:
                    search_result = EvolutionarySearchResult(
                        best_candidate=overall_best,
                        generation_history=generation_history,
                        total_evaluations=evaluator.total_evaluations,
                        unique_architectures_evaluated=len(evaluator.cache),
                        cache_hits=evaluator.cache_hits,
                        total_runtime_seconds=round(total_time, 2),
                        converged=False,
                        stop_reason="max_generations_reached",
                        experiment_run_id=current_job.experiment_run_id,
                    )
                    cls.persist_search_artifacts(search_result, overall_best, current_job)
                    with current_job.lock:
                        current_job.persistence_status = "persisted"
                    logger.info("Persisted UI search trajectory and best architecture successfully.")
                except Exception as pe:
                    with current_job.lock:
                        current_job.persistence_status = "failed"
                        current_job.persistence_error = str(pe)
                    logger.error("Could not persist UI search results: %s", pe)

        except Exception as e:
            logger.error("Error in evolutionary search worker: %s", e)
            with current_job.lock:
                current_job.status = "error"
                current_job.persistence_status = "not_persisted"
                current_job.error_message = str(e)


