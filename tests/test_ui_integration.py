"""Integration tests for UI service interactions and workflow end-to-end integrity."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from src.quantum.architecture_schema import ArchitectureConfig
from ui.services.benchmark_service import BenchmarkService
from ui.services.circuit_service import CircuitService
from ui.services.dataset_service import DatasetService
from ui.services.experiment_service import ExperimentService
from ui.services.noise_service import NoiseService


class TestUIIntegrationWorkflows:
    def test_pipeline_zero_leakage_flow(self):
        """Verify that step-by-step dataset pipeline transformations work on test split."""
        step_outputs = DatasetService.inspect_sample_pipeline(sample_idx=5, split="test")

        # Raw 28x28 normalized to 784
        assert step_outputs["raw_image"].shape == (28, 28)
        assert step_outputs["normalized_flat"].shape == (784,)
        # PCA projection to 4 components
        assert step_outputs["pca_features"].shape == (4,)
        # Angle encoding scales to [0, pi]
        angles = step_outputs["scaled_angles"]
        assert angles.shape == (4,)
        assert (angles >= 0.0).all()
        assert (angles <= 3.1416).all()

    def test_circuit_compilation_all_topologies(self):
        """Verify that all entanglement topologies supported compile valid Qiskit circuits."""
        topologies = ["none", "linear", "circular", "pairwise"]

        for topo in topologies:
            config = ArchitectureConfig(
                num_qubits=4,
                num_layers=1,
                rotation_gates=("ry",),
                entanglement_type=topo,
                entanglement_freq=1,
            )
            circuit, meta = CircuitService.compile_from_config(config)
            assert isinstance(circuit, QuantumCircuit)
            assert meta.num_qubits == 4
            assert meta.circuit_depth >= 1
            if topo == "none":
                assert meta.two_qubit_gates == 0

    def test_all_experiment_artifacts_loadable(self):
        """Verify that all experimental records in experiments/results can be parsed."""
        available = ExperimentService.list_saved_experiments()
        assert len(available) >= 4

        for exp in available:
            payload = ExperimentService.load_raw_experiment_file(exp["file_path"])
            assert payload is not None
            assert isinstance(payload, dict)
            assert len(payload) > 0

    def test_benchmark_data_consistency(self):
        """Verify that benchmark data contains 7 models and expected values."""
        records = BenchmarkService.get_records()
        assert len(records) == 7

        opt_vqc = next(
            (r for r in records if "Optimized VQC" in r["model_name"]), None
        )
        assert opt_vqc is not None
        assert opt_vqc["circuit_depth"] > 0
        assert opt_vqc["test_accuracy"] > 0.8

    def test_noise_tiers_consistency(self):
        """Verify that noise tiers match expected physical noise parameter scalings."""
        tiers = NoiseService.get_canonical_tiers()
        assert tiers["Low Noise"].single_qubit_error < tiers["Medium Noise"].single_qubit_error
        assert tiers["Medium Noise"].single_qubit_error < tiers["High Noise"].single_qubit_error
        assert tiers["Low Noise"].two_qubit_error < tiers["Medium Noise"].two_qubit_error
        assert tiers["Medium Noise"].two_qubit_error < tiers["High Noise"].two_qubit_error
