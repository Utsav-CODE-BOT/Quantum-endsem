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
