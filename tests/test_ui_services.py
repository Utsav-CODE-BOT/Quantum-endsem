"""Unit tests for UI service layer adapters."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from src.quantum.architecture_schema import ArchitectureConfig
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
