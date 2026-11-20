"""Unit tests for UI service layer adapters."""

from __future__ import annotations

import json
import pytest
from qiskit import QuantumCircuit

from src.evaluation.evaluator import CandidateEvaluationResult
from src.evaluation.fitness import FitnessBreakdown
from src.optimization.evolutionary_search import EvolutionarySearchResult, GenerationRecord
from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.circuit_generator import CircuitMetadata
from src.evaluation.classification_metrics import ClassificationReport
from ui.services.benchmark_service import BenchmarkService
from ui.services.circuit_service import CircuitService
from ui.services.dataset_service import DatasetService
from ui.services.experiment_service import ExperimentService
from ui.services.noise_service import NoiseService
from ui.services.search_service import SearchJobState, SearchService


class TestDatasetService:
    def test_get_dataset_summary(self):
        summary = DatasetService.get_dataset_summary()
        assert "Fashion-MNIST" in summary["name"]
        assert summary["train_samples"] == 200
        assert summary["val_samples"] == 60
        assert summary["test_samples"] == 100
        assert summary["total_preprocessed"] == 360
        assert summary["num_pca_components"] == 4
        assert len(summary["explained_variance_ratio"]) == 4

    def test_get_quantum_ready_data(self):
        data = DatasetService.get_quantum_ready_data()
        assert "X_train" in data
        assert "y_train" in data
        assert "X_val" in data
        assert "y_val" in data
        assert "X_test" in data
        assert "y_test" in data
        assert data["X_train"].shape == (200, 4)
        assert data["y_train"].shape == (200,)

    def test_inspect_sample_pipeline(self):
        sample = DatasetService.inspect_sample_pipeline(sample_idx=0, split="train")
        assert sample["index"] == 0
        assert sample["raw_image"].shape == (28, 28)
        assert sample["normalized_flat"].shape == (784,)
        assert sample["pca_features"].shape == (4,)
        assert sample["scaled_angles"].shape == (4,)
        assert sample["class_name"] in ["T-shirt/top", "Ankle boot"]


class TestCircuitService:
    def test_get_canonical_architectures(self):
        presets = CircuitService.get_canonical_architectures()
        assert "Search-Optimized (arch_6327d7bcb8)" in presets
        assert "Baseline Standard VQC" in presets
        assert "Fixed Shallow VQC" in presets
        assert "Fixed Deep VQC" in presets

    def test_compile_from_config(self):
        config = ArchitectureConfig(
            num_qubits=4,
            num_layers=1,
            rotation_gates=("rx", "ry"),
            entanglement_type="circular",
            entanglement_freq=2,
        )
        circuit, meta = CircuitService.compile_from_config(config)
        assert isinstance(circuit, QuantumCircuit)
        assert meta.num_qubits == 4
        assert meta.circuit_depth == 5
        assert meta.num_trainable_parameters == 16
        assert meta.two_qubit_gates == 0

    def test_draw_text(self):
        config = ArchitectureConfig(
            num_qubits=2,
            num_layers=1,
            rotation_gates=("ry",),
            entanglement_type="none",
            entanglement_freq=1,
        )
        circuit, _ = CircuitService.compile_from_config(config)
        ascii_text = CircuitService.draw_text(circuit)
        assert "q_0:" in ascii_text or "q:" in ascii_text

    def test_export_circuit_png_bytes(self):
        config = ArchitectureConfig(
            num_qubits=2,
            num_layers=1,
            rotation_gates=("ry",),
            entanglement_type="none",
            entanglement_freq=1,
        )
        circuit, _ = CircuitService.compile_from_config(config)
        png_bytes = CircuitService.export_circuit_png_bytes(circuit)
        assert len(png_bytes) > 100
        assert png_bytes[:4] == b"\x89PNG"


class TestSearchService:
    def test_load_saved_history(self):
        history = SearchService.load_saved_history()
        assert "generation_history" in history
        assert "best_candidate" in history
        assert len(history["generation_history"]) >= 1

    def test_load_saved_best_architecture(self):
        best = SearchService.load_saved_best_architecture()
        assert "arch_id" in best
        assert best["circuit_metadata"]["circuit_depth"] > 0
        assert "two_qubit_gates" in best["circuit_metadata"]

    def test_search_job_state_thread_safety(self):
        state = SearchJobState()
        assert state.status == "idle"
        status_dict = state.to_dict()
        assert status_dict["status"] == "idle"
        assert status_dict["best_fitness"] == 0.0

    def test_cancel_search_request(self):
        SearchService._active_job = SearchJobState()
        SearchService._active_job.status = "running"
        SearchService.cancel_search()
        assert SearchService._active_job._cancel_requested is True
        assert SearchService._active_job.status == "cancelling"

    def test_start_search_concurrency_guard(self):
        SearchService._active_job = SearchJobState()
        SearchService._active_job.status = "running"
        started = SearchService.start_search(pop_size=2, generations=1)
        assert started is False
        SearchService._active_job.status = "idle"

    def test_actual_persistence_behavior(self, tmp_path):
        """Verify production SearchService.persist_search_artifacts writes valid files to disk."""
        cfg = ArchitectureConfig(num_qubits=4, num_layers=1, rotation_gates=("ry",), entanglement_type="none")
        meta = CircuitMetadata(
            num_qubits=4, circuit_depth=5, total_gates=10, single_qubit_gates=10,
            two_qubit_gates=0, gate_distribution={"ry": 10}, num_input_features=4, num_trainable_parameters=8,
        )
        report = ClassificationReport(
            accuracy=0.95, precision_macro=0.95, recall_macro=0.95, f1_macro=0.95,
            confusion_matrix=[[28, 2], [1, 29]], roc_auc=0.99, num_samples=60,
        )
        fitness_bd = FitnessBreakdown(
            scalar_fitness=0.65, performance_score=0.95, complexity_score=0.05,
            robustness_score=1.0, w_performance=0.5, w_complexity=0.3, w_robustness=0.2,
        )
        cand = CandidateEvaluationResult(
            arch_id="test_cand_prod", config=cfg, circuit_metadata=meta,
            val_metrics=report, fitness=0.65, fitness_breakdown=fitness_bd,
            training_time_seconds=1.2, evaluation_time_seconds=0.1, is_cached=False,
            success=True,
        )
        gen_rec = GenerationRecord(
            generation=1, best_candidate=cand, best_fitness=0.65, avg_fitness=0.65,
            candidates=[cand], elapsed_time_seconds=1.3,
        )
        search_result = EvolutionarySearchResult(
            best_candidate=cand,
            generation_history=[gen_rec],
            total_evaluations=1,
            unique_architectures_evaluated=1,
            cache_hits=0,
            total_runtime_seconds=1.5,
            converged=False,
            stop_reason="max_generations_reached",
            experiment_run_id="EXP-PROD-TEST",
        )
        job = SearchJobState()
        job.experiment_run_id = "EXP-PROD-TEST"

        hist_file = tmp_path / "search_history.json"
        best_file = tmp_path / "best_architecture.json"

        updated = SearchService.persist_search_artifacts(
            search_result=search_result,
            overall_best=cand,
            current_job=job,
            hist_path=hist_file,
            best_arch_path=best_file,
        )
        assert updated is True
        assert hist_file.exists()
        assert best_file.exists()

        with open(best_file, "r", encoding="utf-8") as f:
            best_data = json.load(f)
        assert best_data["arch_id"] == "test_cand_prod"
        assert best_data["experiment_run_id"] == "EXP-PROD-TEST"

    def test_filesystem_write_failure_handling(self, tmp_path, monkeypatch):
        """Verify production worker catches write failure and records persistence_status='failed'."""
        cfg = ArchitectureConfig(num_qubits=4, num_layers=1, rotation_gates=("ry",), entanglement_type="none")
        meta = CircuitMetadata(
            num_qubits=4, circuit_depth=5, total_gates=10, single_qubit_gates=10,
            two_qubit_gates=0, gate_distribution={"ry": 10}, num_input_features=4, num_trainable_parameters=8,
        )
        report = ClassificationReport(
            accuracy=0.95, precision_macro=0.95, recall_macro=0.95, f1_macro=0.95,
            confusion_matrix=[[28, 2], [1, 29]], roc_auc=0.99, num_samples=60,
        )
        fitness_bd = FitnessBreakdown(
            scalar_fitness=0.65, performance_score=0.95, complexity_score=0.05,
            robustness_score=1.0, w_performance=0.5, w_complexity=0.3, w_robustness=0.2,
        )
        cand = CandidateEvaluationResult(
            arch_id="test_cand_fail", config=cfg, circuit_metadata=meta,
            val_metrics=report, fitness=0.65, fitness_breakdown=fitness_bd,
            training_time_seconds=0.1, evaluation_time_seconds=0.1, is_cached=False,
            success=True,
        )
        job = SearchJobState()
        job.experiment_run_id = "EXP-FAIL-TEST"

        def mock_failing_persist(*args, **kwargs):
            raise OSError("Simulated I/O permission denied writing artifacts")

        monkeypatch.setattr(SearchService, "persist_search_artifacts", mock_failing_persist)
        monkeypatch.setattr("src.evaluation.evaluator.ArchitectureEvaluator.evaluate", lambda *a, **k: cand)

        SearchService._run_worker(
            current_job=job,
            pop_size=2,
            generations=1,
            maxiter=1,
            w_perf=0.5,
            w_comp=0.3,
            w_rob=0.2,
            crossover_rate=0.7,
            mutation_rate=0.3,
            tournament_size=2,
            random_seed=42,
        )

        status = job.to_dict()
        assert status["status"] == "completed"
        assert status["persistence_status"] == "failed"
        assert "Simulated I/O permission denied" in status["persistence_error"]

    def test_search_cancellation_production(self, monkeypatch):
        """Verify production SearchService.cancel_search transitions job to cancelled and not_persisted."""
        import time
        SearchService._active_job = SearchJobState()

        # Mock evaluate to simulate brief evaluation work
        cfg = ArchitectureConfig(num_qubits=4, num_layers=1, rotation_gates=("ry",), entanglement_type="none")
        meta = CircuitMetadata(num_qubits=4, circuit_depth=5, total_gates=10, single_qubit_gates=10, two_qubit_gates=0, gate_distribution={}, num_input_features=4, num_trainable_parameters=8)
        rep = ClassificationReport(accuracy=0.9, precision_macro=0.9, recall_macro=0.9, f1_macro=0.9, confusion_matrix=[[27, 3], [3, 27]], roc_auc=0.95, num_samples=60)
        fbd = FitnessBreakdown(scalar_fitness=0.6, performance_score=0.9, complexity_score=0.1, robustness_score=1.0, w_performance=0.5, w_complexity=0.3, w_robustness=0.2)
        mock_cand = CandidateEvaluationResult(arch_id="cand_slow", config=cfg, circuit_metadata=meta, val_metrics=rep, fitness=0.6, fitness_breakdown=fbd, training_time_seconds=0.01, evaluation_time_seconds=0.01, is_cached=False, success=True)

        def slow_eval(*args, **kwargs):
            time.sleep(0.05)
            return mock_cand

        monkeypatch.setattr("src.evaluation.evaluator.ArchitectureEvaluator.evaluate", slow_eval)

        SearchService.start_search(pop_size=4, generations=5, maxiter=1)
        time.sleep(0.02)

        # Production cancellation call
        SearchService.cancel_search()

        # Wait for worker thread to exit cooperatively
        for _ in range(50):
            if not SearchService.is_running():
                break
            time.sleep(0.05)

        status = SearchService.get_status()
        assert status["status"] == "cancelled"
        assert status["persistence_status"] == "not_persisted"

    def test_failed_candidate_exclusion_in_worker(self, monkeypatch):
        """Verify production worker excludes failed candidates and transitions to error if none succeed."""
        cfg = ArchitectureConfig(num_qubits=4, num_layers=1, rotation_gates=("ry",), entanglement_type="none")
        meta = CircuitMetadata(
            num_qubits=4, circuit_depth=5, total_gates=10, single_qubit_gates=10,
            two_qubit_gates=0, gate_distribution={}, num_input_features=4, num_trainable_parameters=8,
        )
        report = ClassificationReport(
            accuracy=0.0, precision_macro=0.0, recall_macro=0.0, f1_macro=0.0,
            confusion_matrix=[[0, 0], [0, 0]], roc_auc=0.0, num_samples=0,
        )
        fitness_bd = FitnessBreakdown(
            scalar_fitness=-1.0, performance_score=0.0, complexity_score=0.0,
            robustness_score=0.0, w_performance=0.5, w_complexity=0.3, w_robustness=0.2,
        )
        failed_cand = CandidateEvaluationResult(
            arch_id="failed_cand", config=cfg, circuit_metadata=meta,
            val_metrics=report, fitness=-1.0, fitness_breakdown=fitness_bd,
            training_time_seconds=0.1, evaluation_time_seconds=0.1, is_cached=False,
            success=False, error_message="Simulated quantum execution failure",
        )

        job = SearchJobState()
        job.experiment_run_id = "EXP-FAIL-POP"

        # Mock evaluate to return only failed candidate
        monkeypatch.setattr("src.evaluation.evaluator.ArchitectureEvaluator.evaluate", lambda *a, **k: failed_cand)

        SearchService._run_worker(
            current_job=job,
            pop_size=2,
            generations=1,
            maxiter=1,
            w_perf=0.5,
            w_comp=0.3,
            w_rob=0.2,
            crossover_rate=0.7,
            mutation_rate=0.3,
            tournament_size=2,
            random_seed=42,
        )

        status = job.to_dict()
        assert status["status"] == "error"
        assert status["persistence_status"] == "not_persisted"
        assert "All candidate evaluations failed in generation 1" in status["error_message"]

    def test_inferior_candidate_cannot_overwrite_existing_best(self, tmp_path):
        """Verify production SearchService.persist_search_artifacts guards higher-fitness best architecture."""
        best_arch_file = tmp_path / "best_architecture.json"
        existing_best = {
            "arch_id": "existing_high_performer",
            "fitness": 0.95,
            "val_metrics": {"accuracy": 0.98},
            "circuit_metadata": {"circuit_depth": 5, "two_qubit_gates": 0},
        }
        with open(best_arch_file, "w", encoding="utf-8") as f:
            json.dump(existing_best, f, indent=2)

        cfg = ArchitectureConfig(num_qubits=4, num_layers=1, rotation_gates=("ry",), entanglement_type="none")
        meta = CircuitMetadata(
            num_qubits=4, circuit_depth=5, total_gates=10, single_qubit_gates=10,
            two_qubit_gates=0, gate_distribution={"ry": 10}, num_input_features=4, num_trainable_parameters=8,
        )
        report = ClassificationReport(
            accuracy=0.60, precision_macro=0.60, recall_macro=0.60, f1_macro=0.60,
            confusion_matrix=[[18, 12], [12, 18]], roc_auc=0.60, num_samples=60,
        )
        fitness_bd = FitnessBreakdown(
            scalar_fitness=0.45, performance_score=0.60, complexity_score=0.15,
            robustness_score=1.0, w_performance=0.5, w_complexity=0.3, w_robustness=0.2,
        )
        inferior_cand = CandidateEvaluationResult(
            arch_id="inferior_cand", config=cfg, circuit_metadata=meta,
            val_metrics=report, fitness=0.45, fitness_breakdown=fitness_bd,
            training_time_seconds=1.0, evaluation_time_seconds=0.1, is_cached=False,
            success=True,
        )
        gen_rec = GenerationRecord(
            generation=1, best_candidate=inferior_cand, best_fitness=0.45, avg_fitness=0.45,
            candidates=[inferior_cand], elapsed_time_seconds=1.1,
        )
        search_result = EvolutionarySearchResult(
            best_candidate=inferior_cand,
            generation_history=[gen_rec],
            total_evaluations=1,
            unique_architectures_evaluated=1,
            cache_hits=0,
            total_runtime_seconds=1.2,
            converged=False,
            stop_reason="max_generations_reached",
            experiment_run_id="EXP-INFERIOR",
        )
        job = SearchJobState()
        job.experiment_run_id = "EXP-INFERIOR"

        hist_file = tmp_path / "search_history.json"
        updated = SearchService.persist_search_artifacts(
            search_result=search_result,
            overall_best=inferior_cand,
            current_job=job,
            hist_path=hist_file,
            best_arch_path=best_arch_file,
        )

        assert updated is False

        # Read back and confirm higher fitness candidate was preserved
        with open(best_arch_file, "r", encoding="utf-8") as f:
            reloaded = json.load(f)
        assert reloaded["arch_id"] == "existing_high_performer"
        assert reloaded["fitness"] == 0.95


class TestNoiseService:
    def test_get_canonical_tiers(self):
        tiers = NoiseService.get_canonical_tiers()
        assert "Low Noise" in tiers
        assert "Medium Noise" in tiers
        assert "High Noise" in tiers
        assert tiers["High Noise"].two_qubit_error > tiers["Low Noise"].two_qubit_error

    def test_load_saved_results(self):
        results = NoiseService.load_saved_results()
        assert "benchmarks" in results
        assert "baseline" in results["benchmarks"]
        assert "optimized" in results["benchmarks"]
        assert len(results["benchmarks"]["baseline"]) == 3


class TestBenchmarkService:
    def test_load_benchmark_data(self):
        data = BenchmarkService.load_benchmark_data()
        assert "results" in data
        assert len(data["results"]) == 7

    def test_get_records(self):
        records = BenchmarkService.get_records()
        assert len(records) == 7
        names = [r["model_name"] for r in records]
        assert "Linear SVM" in names
        assert any("Optimized VQC" in n for n in names)
        assert "Fixed Standard VQC" in names

    def test_get_model_by_name(self):
        records = BenchmarkService.get_records()
        opt_name = next(r["model_name"] for r in records if "Optimized VQC" in r["model_name"])
        model = BenchmarkService.get_model_by_name(opt_name)
        assert model is not None
        assert model["test_accuracy"] > 0.8
        assert model["circuit_depth"] > 0


class TestExperimentService:
    def test_list_saved_experiments(self):
        experiments = ExperimentService.list_saved_experiments()
        assert len(experiments) >= 4
        exp_ids = [e["experiment_id"] for e in experiments]
        assert "exp_baseline_vqc_depth12" in exp_ids
        assert "exp_evolutionary_search_run1" in exp_ids
        assert "exp_noise_robustness_nisq" in exp_ids

    def test_load_baseline_results(self):
        baseline = ExperimentService.load_baseline_results()
        assert "circuit_metadata" in baseline
        assert baseline["circuit_metadata"]["circuit_depth"] > 0
        assert "validation_metrics" in baseline

    def test_load_raw_experiment_file(self):
        experiments = ExperimentService.list_saved_experiments()
        first_path = experiments[0]["file_path"]
        raw = ExperimentService.load_raw_experiment_file(first_path)
        assert isinstance(raw, dict)
        assert len(raw) > 0

    def test_export_records_to_csv(self):
        experiments = ExperimentService.list_saved_experiments()
        csv_str = ExperimentService.export_records_to_csv(experiments)
        assert "experiment_id" in csv_str
        assert "exp_baseline_vqc_depth12" in csv_str
