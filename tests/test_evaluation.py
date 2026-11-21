"""Unit tests for FitnessCalculator and ArchitectureEvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.evaluator import ArchitectureEvaluator, CandidateEvaluationResult
from src.evaluation.fitness import FitnessCalculator
from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.circuit_generator import CircuitMetadata


class TestFitnessCalculator:
    """Test normalized scalar multi-objective fitness calculation."""

    def test_fitness_bounds_and_breakdown(self) -> None:
        calc = FitnessCalculator(
            w_performance=0.5,
            w_complexity=0.3,
            w_robustness=0.2,
            max_depth=50,
            max_cx_gates=30,
        )
        meta = CircuitMetadata(
            num_qubits=4,
            circuit_depth=10,
            total_gates=20,
            single_qubit_gates=16,
            two_qubit_gates=4,
            gate_distribution={"ry": 8, "rz": 8, "cx": 4},
            num_input_features=4,
            num_trainable_parameters=16,
        )

        breakdown = calc.evaluate(val_accuracy=0.9, metadata=meta)
        assert 0.0 <= breakdown.performance_score <= 1.0
        assert 0.0 <= breakdown.complexity_score <= 1.0
        assert 0.0 <= breakdown.robustness_score <= 1.0
        assert breakdown.scalar_fitness > 0.0

    def test_complexity_penalty_increases_with_depth(self) -> None:
        calc = FitnessCalculator(max_depth=50, max_cx_gates=30)
        shallow_meta = CircuitMetadata(
            num_qubits=4, circuit_depth=5, total_gates=10,
            single_qubit_gates=8, two_qubit_gates=2, gate_distribution={},
            num_input_features=4, num_trainable_parameters=8,
        )
        deep_meta = CircuitMetadata(
            num_qubits=4, circuit_depth=35, total_gates=40,
            single_qubit_gates=25, two_qubit_gates=15, gate_distribution={},
            num_input_features=4, num_trainable_parameters=30,
        )
        comp_shallow = calc.calculate_complexity(shallow_meta)
        comp_deep = calc.calculate_complexity(deep_meta)
        assert comp_deep > comp_shallow


class TestArchitectureEvaluator:
    """Test candidate architecture evaluation and caching."""

    @pytest.fixture
    def small_splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        return X_train, y_train, X_val, y_val

    def test_evaluator_execution_and_caching(self, small_splits) -> None:
        X_train, y_train, X_val, y_val = small_splits
        evaluator = ArchitectureEvaluator(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            maxiter=10,
            enable_cache=True,
        )

        config = ArchitectureConfig(
            num_qubits=4,
            num_layers=1,
            rotation_gates=("ry",),
            entanglement_type="linear",
        )

        # 1. First evaluation: fresh training
        res1 = evaluator.evaluate(config)
        assert isinstance(res1, CandidateEvaluationResult)
        assert res1.success is True
        assert not res1.is_cached
        assert res1.fitness_breakdown is not None
        assert evaluator.cache_hits == 0

        # 2. Second evaluation: cache hit
        res2 = evaluator.evaluate(config)
        assert res2.is_cached is True
        assert evaluator.cache_hits == 1
        assert res1.fitness == res2.fitness

    def test_cache_key_discrimination_across_conditions(self, small_splits) -> None:
        """Verify cache keys correctly distinguish dataset fingerprints, budgets, noise, and architectures."""
        X_train, y_train, X_val, y_val = small_splits
        cfg1 = ArchitectureConfig(4, 1, ("ry",), "linear")
        cfg2 = ArchitectureConfig(4, 2, ("rx",), "circular")

        base_eval = ArchitectureEvaluator(X_train, y_train, X_val, y_val, maxiter=10)
        key_base = base_eval._generate_cache_key(cfg1)

        # 1. Same conditions produce identical key
        key_same = base_eval._generate_cache_key(cfg1)
        assert key_base == key_same

        # 2. Different architecture configuration produces different key
        key_diff_arch = base_eval._generate_cache_key(cfg2)
        assert key_base != key_diff_arch

        # 3. Different training data produces different key
        eval_diff_train = ArchitectureEvaluator(X_train + 0.1, y_train, X_val, y_val, maxiter=10)
        assert key_base != eval_diff_train._generate_cache_key(cfg1)

        # 4. Different validation data produces different key
        eval_diff_val = ArchitectureEvaluator(X_train, y_train, X_val + 0.1, y_val, maxiter=10)
        assert key_base != eval_diff_val._generate_cache_key(cfg1)

        # 5. Different optimization budget produces different key
        eval_diff_iter = ArchitectureEvaluator(X_train, y_train, X_val, y_val, maxiter=25)
        assert key_base != eval_diff_iter._generate_cache_key(cfg1)

        # 6. Different noise evaluation setting produces different key
        eval_no_noise = ArchitectureEvaluator(X_train, y_train, X_val, y_val, maxiter=10, evaluate_noise=False)
        assert key_base != eval_no_noise._generate_cache_key(cfg1)

        # 7. Different fitness objective weights produce different key
        diff_fitness = FitnessCalculator(w_performance=0.8, w_complexity=0.1, w_robustness=0.1)
        eval_diff_fit = ArchitectureEvaluator(X_train, y_train, X_val, y_val, fitness_calculator=diff_fitness, maxiter=10)
        assert key_base != eval_diff_fit._generate_cache_key(cfg1)

    def test_evaluator_active_noise_robustness(self, small_splits) -> None:
        X_train, y_train, X_val, y_val = small_splits
        evaluator = ArchitectureEvaluator(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            maxiter=5,
            evaluate_noise=True,
            enable_cache=False,
        )
        config = ArchitectureConfig(
            num_qubits=4,
            num_layers=1,
            rotation_gates=("ry",),
            entanglement_type="linear",
        )
        res = evaluator.evaluate(config)
        assert res.success is True
        assert res.noisy_val_metrics is not None
        assert 0.0 <= res.noisy_val_metrics.accuracy <= 1.0
        assert 0.0 <= res.fitness_breakdown.robustness_score <= 1.0

    def test_fitness_calculator_missing_noise_zero_score(self) -> None:
        """Verify missing or None noisy_accuracy receives 0.0 robustness, never rewarded."""
        calc = FitnessCalculator()
        robustness = calc.calculate_robustness(ideal_accuracy=0.95, noisy_accuracy=None)
        assert robustness == 0.0

    def test_robustness_bounded_between_zero_and_one(self) -> None:
        """Verify R(A) = min(1.0, max(0.0, Acc_noisy / Acc_val)) upper and lower safeguards."""
        calc = FitnessCalculator()
        # Explicit Phase 1 specification table
        assert calc.calculate_robustness(ideal_accuracy=0.90, noisy_accuracy=0.90) == 1.0
        assert calc.calculate_robustness(ideal_accuracy=0.90, noisy_accuracy=0.81) == pytest.approx(0.90)
        assert calc.calculate_robustness(ideal_accuracy=0.90, noisy_accuracy=0.95) == 1.0
        assert calc.calculate_robustness(ideal_accuracy=0.00, noisy_accuracy=0.00) == 0.0
        assert calc.calculate_robustness(ideal_accuracy=0.00, noisy_accuracy=0.01) == 0.0
        assert calc.calculate_robustness(ideal_accuracy=0.90, noisy_accuracy=None) == 0.0

        # Additional edge-case safeguards
        assert calc.calculate_robustness(ideal_accuracy=0.80, noisy_accuracy=0.0) == 0.0
        assert calc.calculate_robustness(ideal_accuracy=0.0, noisy_accuracy=0.50) == 0.0
        assert calc.calculate_robustness(ideal_accuracy=-0.5, noisy_accuracy=0.5) == 0.0
        assert calc.calculate_robustness(ideal_accuracy=0.5, noisy_accuracy=-0.2) == 0.0
        assert calc.calculate_robustness(ideal_accuracy=float("nan"), noisy_accuracy=0.5) == 0.0
        assert calc.calculate_robustness(ideal_accuracy=0.5, noisy_accuracy=float("inf")) == 0.0

    def test_complexity_edge_cases_and_safeguards(self) -> None:
        """Verify complexity formula bounds, edge cases, and normalization clamps."""
        calc = FitnessCalculator(max_depth=50, max_cx_gates=30)
        # Zero gates and zero depth -> complexity 0.0
        zero_meta = CircuitMetadata(
            num_qubits=4, circuit_depth=0, total_gates=0,
            single_qubit_gates=0, two_qubit_gates=0, gate_distribution={},
            num_input_features=4, num_trainable_parameters=0,
        )
        assert calc.calculate_complexity(zero_meta) == 0.0

        # Normalization upper limit clamps to 1.0
        huge_meta = CircuitMetadata(
            num_qubits=4, circuit_depth=100, total_gates=200,
            single_qubit_gates=100, two_qubit_gates=100, gate_distribution={},
            num_input_features=4, num_trainable_parameters=200,
        )
        assert calc.calculate_complexity(huge_meta) == 1.0

        # Non-finite values return 1.0 penalty
        nan_meta = CircuitMetadata(
            num_qubits=4, circuit_depth=float("nan"), total_gates=10,
            single_qubit_gates=5, two_qubit_gates=5, gate_distribution={},
            num_input_features=4, num_trainable_parameters=10,
        )
        assert calc.calculate_complexity(nan_meta) == 1.0

    def test_evaluator_disqualifies_candidate_on_noise_failure(self, small_splits, monkeypatch) -> None:
        """Verify that when noise evaluation fails, candidate is disqualified (success=False, fitness=-1.0)."""
        from src.evaluation.noise_evaluator import NoiseEvaluator

        def mock_failing_noise_evaluate(*args, **kwargs):
            raise RuntimeError("Simulated Aer hardware failure")

        monkeypatch.setattr(NoiseEvaluator, "evaluate", mock_failing_noise_evaluate)

        X_train, y_train, X_val, y_val = small_splits
        evaluator = ArchitectureEvaluator(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            maxiter=5,
            evaluate_noise=True,
            enable_cache=False,
        )
        config = ArchitectureConfig(
            num_qubits=4,
            num_layers=1,
            rotation_gates=("ry",),
            entanglement_type="none",
        )
        res = evaluator.evaluate(config)
        assert res.success is False
        assert res.fitness == -1.0
        assert "Simulated Aer hardware failure" in (res.error_message or "")
