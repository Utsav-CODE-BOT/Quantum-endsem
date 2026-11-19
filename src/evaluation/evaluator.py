"""Candidate VQC architecture evaluator with caching and resource tracking."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from src.evaluation.classification_metrics import (
    ClassificationReport,
    compute_classification_metrics,
)
from src.evaluation.fitness import FitnessBreakdown, FitnessCalculator
from src.evaluation.noise_evaluator import NoiseEvaluator, NoiseModelConfig
from src.quantum.architecture_schema import ArchitectureConfig
from src.quantum.circuit_generator import CircuitMetadata
from src.quantum.vqc_model import ConfigurableVQC
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CandidateEvaluationResult:
    """Standardized result of evaluating a single VQC architecture candidate."""

    arch_id: str
    config: ArchitectureConfig
    circuit_metadata: CircuitMetadata
    val_metrics: ClassificationReport
    train_metrics: ClassificationReport | None = None
    test_metrics: ClassificationReport | None = None
    noisy_val_metrics: ClassificationReport | None = None
    fitness_breakdown: FitnessBreakdown | None = None
    fitness: float = 0.0
    training_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0
    is_cached: bool = False
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON persistence."""
        return {
            "arch_id": self.arch_id,
            "config": self.config.to_dict(),
            "circuit_metadata": self.circuit_metadata.to_dict(),
            "val_metrics": self.val_metrics.to_dict(),
            "train_metrics": self.train_metrics.to_dict() if self.train_metrics else None,
            "test_metrics": self.test_metrics.to_dict() if self.test_metrics else None,
            "noisy_val_metrics": self.noisy_val_metrics.to_dict() if self.noisy_val_metrics else None,
            "fitness": self.fitness,
            "fitness_breakdown": self.fitness_breakdown.to_dict() if self.fitness_breakdown else None,
            "training_time_seconds": self.training_time_seconds,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "is_cached": self.is_cached,
            "success": self.success,
            "error_message": self.error_message,
        }


class ArchitectureEvaluator:
    """Trains and evaluates candidate VQC architectures on validation data."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        fitness_calculator: FitnessCalculator | None = None,
        maxiter: int = 35,
        optimizer_name: str = "cobyla",
        training_seed: int = 42,
        enable_cache: bool = True,
        noise_config: NoiseModelConfig | None = None,
        evaluate_noise: bool = True,
    ) -> None:
        """Initialize evaluator.

        Args:
            X_train: Training features (N_train, n_qubits).
            y_train: Training binary labels (N_train,).
            X_val: Validation features (N_val, n_qubits).
            y_val: Validation binary labels (N_val,).
            fitness_calculator: FitnessCalculator instance.
            maxiter: Optimizer iterations per candidate evaluation.
            optimizer_name: Classical optimizer name.
            training_seed: Seed for weight initialization.
            enable_cache: Whether to cache evaluation results.
            noise_config: Noise model parameters for validation robustness scoring.
            evaluate_noise: Whether to evaluate noise robustness on validation data.
        """
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=int)
        self.X_val = np.asarray(X_val, dtype=float)
        self.y_val = np.asarray(y_val, dtype=int)

        self.fitness_calculator = fitness_calculator or FitnessCalculator()
        self.maxiter = maxiter
        self.optimizer_name = optimizer_name
        self.training_seed = training_seed
        self.enable_cache = enable_cache

        self.noise_config = noise_config or NoiseModelConfig(
            name="val_nisq_noise",
            single_qubit_error=0.001,
            two_qubit_error=0.01,
            measurement_error=0.015,
            shots=512,
        )
        self.evaluate_noise = evaluate_noise

        # Cache mapping unique evaluation key -> CandidateEvaluationResult
        self.cache: dict[str, CandidateEvaluationResult] = {}
        self.total_evaluations = 0
        self.cache_hits = 0

    def _generate_cache_key(self, config: ArchitectureConfig) -> str:
        """Create cache key accounting for architecture and training conditions."""
        key_str = (
            f"{config.arch_id}:opt={self.optimizer_name}:iter={self.maxiter}:"
            f"seed={self.training_seed}:N={len(self.X_train)}:noise={self.evaluate_noise}"
        )
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def evaluate(self, config: ArchitectureConfig) -> CandidateEvaluationResult:
        """Train and evaluate a single architecture candidate on validation data.

        Strict Scientific Protocol:
        - Candidate selection is driven entirely by validation metrics (accuracy + noise robustness).
        - Test data is NEVER used or exposed during architecture evaluation.

        Args:
            config: ArchitectureConfig representing the candidate circuit.

        Returns:
            CandidateEvaluationResult dataclass.
        """
        cache_key = self._generate_cache_key(config)
        if self.enable_cache and cache_key in self.cache:
            self.cache_hits += 1
            cached = self.cache[cache_key]
            logger.debug("Evaluation cache hit for candidate %s (arch_id=%s)", cache_key, config.arch_id)
            # Return copy with is_cached=True
            return CandidateEvaluationResult(
                arch_id=cached.arch_id,
                config=cached.config,
                circuit_metadata=cached.circuit_metadata,
                val_metrics=cached.val_metrics,
                train_metrics=cached.train_metrics,
                noisy_val_metrics=cached.noisy_val_metrics,
                fitness_breakdown=cached.fitness_breakdown,
                fitness=cached.fitness,
                training_time_seconds=cached.training_time_seconds,
                evaluation_time_seconds=cached.evaluation_time_seconds,
                is_cached=True,
                success=cached.success,
            )

        self.total_evaluations += 1
        t_start = time.perf_counter()

        try:
            # 1. Build VQC model
            vqc = ConfigurableVQC(
                num_qubits=config.num_qubits,
                num_layers=config.num_layers,
                rotation_gates=config.rotation_gates,
                entanglement_type=config.entanglement_type,
                entanglement_freq=config.entanglement_freq,
                feature_rotation=config.feature_rotation,
                optimizer_name=self.optimizer_name,
                maxiter=self.maxiter,
                random_seed=self.training_seed,
                measurement_strategy=config.measurement_strategy,
            )

            # 2. Train on training partition
            vqc.fit(self.X_train, self.y_train)
            t_trained = time.perf_counter()
            train_time = t_trained - t_start

            # 3. Evaluate on validation partition (Ideal)
            val_preds = vqc.predict(self.X_val)
            val_probs = vqc.predict_proba(self.X_val)
            val_metrics = compute_classification_metrics(self.y_val, val_preds, val_probs)
            t_eval = time.perf_counter() - t_trained

            # 4. Evaluate under simulated NISQ noise on validation partition (3rd Objective)
            noisy_val_metrics = None
            noisy_val_accuracy = None
            if self.evaluate_noise:
                try:
                    noise_evaluator = NoiseEvaluator(seed=self.training_seed)
                    noise_report = noise_evaluator.evaluate(
                        vqc=vqc,
                        X_test=self.X_val,
                        y_test=self.y_val,
                        noise_config=self.noise_config,
                    )
                    noisy_val_metrics = noise_report.noisy_metrics
                    noisy_val_accuracy = noise_report.noisy_metrics.accuracy
                except Exception as ex:
                    logger.warning("Noise evaluation fallback for %s: %s", config.arch_id, ex)
                    noisy_val_accuracy = None

            # 5. Compute multi-objective fitness with active noise robustness
            fitness_breakdown = self.fitness_calculator.evaluate(
                val_accuracy=val_metrics.accuracy,
                metadata=vqc.metadata,
                noisy_accuracy=noisy_val_accuracy,
            )

            result = CandidateEvaluationResult(
                arch_id=config.arch_id,
                config=config,
                circuit_metadata=vqc.metadata,
                val_metrics=val_metrics,
                noisy_val_metrics=noisy_val_metrics,
                fitness_breakdown=fitness_breakdown,
                fitness=fitness_breakdown.scalar_fitness,
                training_time_seconds=round(train_time, 2),
                evaluation_time_seconds=round(t_eval, 4),
                is_cached=False,
                success=True,
            )

        except Exception as e:
            logger.error("Failed evaluation of candidate %s: %s", config.arch_id, e)
            t_total = time.perf_counter() - t_start
            # Construct fallback dummy metadata and minimal fitness
            from src.quantum.circuit_generator import VQCCircuitGenerator
            try:
                gen, meta = VQCCircuitGenerator.from_specs(
                    num_qubits=config.num_qubits,
                    num_layers=config.num_layers,
                    rotation_gates=config.rotation_gates,
                    entanglement_type=config.entanglement_type,
                    entanglement_freq=config.entanglement_freq,
                ), None
                meta = gen.metadata
            except Exception:
                meta = CircuitMetadata(
                    num_qubits=config.num_qubits,
                    circuit_depth=999,
                    total_gates=999,
                    single_qubit_gates=999,
                    two_qubit_gates=999,
                    gate_distribution={},
                    num_input_features=config.num_qubits,
                    num_trainable_parameters=999,
                )

            dummy_metrics = ClassificationReport(
                accuracy=0.0,
                precision_macro=0.0,
                recall_macro=0.0,
                f1_macro=0.0,
                confusion_matrix=[[0, 0], [0, 0]],
                roc_auc=0.5,
            )
            result = CandidateEvaluationResult(
                arch_id=config.arch_id,
                config=config,
                circuit_metadata=meta,
                val_metrics=dummy_metrics,
                fitness=-1.0,
                training_time_seconds=round(t_total, 2),
                is_cached=False,
                success=False,
                error_message=str(e),
            )

        if self.enable_cache:
            self.cache[cache_key] = result

        return result
