"""Configurable Variational Quantum Classifier (VQC) implementation.

Integrates:
- Parameterized Angle Feature Map
- Parameterized Variational Ansatz
- Sampler Primitive (Statevector ideal or Aer noisy)
- Configurable classical optimizers (COBYLA, SPSA, L_BFGS_B)
- Bitstring parity interpretation for binary classification
- Comprehensive training trajectory logging and model checkpointing
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from qiskit.primitives import BaseSampler
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SPSA, Optimizer
from qiskit_machine_learning.algorithms import VQC

from src.quantum.ansatz_builder import AnsatzBuilder
from src.quantum.circuit_generator import CircuitMetadata, VQCCircuitGenerator
from src.quantum.feature_encoding import AngleFeatureMap
from src.quantum.measurement import MeasurementStrategy
from src.quantum.simulator import QuantumSimulatorManager
from src.training.training_history import TrainingHistory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ConfigurableVQC:
    """Configurable Variational Quantum Classifier with scikit-learn compatible interface."""

    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        rotation_gates: Sequence[str] = ("ry", "rz"),
        entanglement_type: str = "linear",
        entanglement_freq: int = 1,
        feature_rotation: str = "ry",
        optimizer_name: str = "cobyla",
        maxiter: int = 50,
        tol: float = 1e-4,
        random_seed: int = 42,
        sampler: BaseSampler | None = None,
        measurement_strategy: str = "parity",
    ) -> None:
        """Initialize the VQC model.

        Args:
            num_qubits: Number of qubits (must match input feature dimension).
            num_layers: Depth of variational ansatz layers.
            rotation_gates: Sequence of rotation gate types per layer.
            entanglement_type: 'linear', 'circular', 'pairwise', or 'none'.
            entanglement_freq: Apply entanglement every k layers.
            feature_rotation: Gate for angle encoding.
            optimizer_name: 'cobyla', 'spsa', or 'l_bfgs_b'.
            maxiter: Maximum classical optimizer iterations.
            tol: Optimization tolerance.
            random_seed: Seed for parameter initialization and optimizer.
            sampler: Qiskit Sampler primitive (defaults to ideal statevector Sampler).
            measurement_strategy: 'parity' or 'single_qubit'.
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.rotation_gates = list(rotation_gates)
        self.entanglement_type = entanglement_type
        self.entanglement_freq = entanglement_freq
        self.feature_rotation = feature_rotation
        self.optimizer_name = optimizer_name.lower()
        self.maxiter = maxiter
        self.tol = tol
        self.random_seed = random_seed
        self.measurement_strategy = measurement_strategy

        # 1. Build circuits
        self.feature_map = AngleFeatureMap(
            num_qubits=num_qubits,
            rotation_gate=feature_rotation,
        )
        self.ansatz = AnsatzBuilder(
            num_qubits=num_qubits,
            num_layers=num_layers,
            rotation_gates=rotation_gates,
            entanglement_type=entanglement_type,
            entanglement_freq=entanglement_freq,
        )
        self.circuit_generator = VQCCircuitGenerator(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
        )
        self.metadata: CircuitMetadata = self.circuit_generator.metadata

        # 2. Simulator & measurement
        self.sampler = sampler or QuantumSimulatorManager.get_ideal_sampler()
        self.interpret_func = MeasurementStrategy.get_interpret_func(measurement_strategy)

        # 3. Optimizer
        self.optimizer = self._build_optimizer()

        # 4. State tracking
        self.training_history = TrainingHistory()
        self._start_time: float = 0.0
        self.vqc_internal: VQC | None = None
        self.is_fitted = False

    def _build_optimizer(self) -> Optimizer:
        """Construct the configured classical optimizer."""
        if self.optimizer_name == "cobyla":
            return COBYLA(maxiter=self.maxiter, tol=self.tol)
        elif self.optimizer_name == "spsa":
            return SPSA(maxiter=self.maxiter)
        elif self.optimizer_name in ("l_bfgs_b", "lbfgsb"):
            return L_BFGS_B(maxiter=self.maxiter)
        else:
            raise ValueError(f"Unsupported optimizer: '{self.optimizer_name}'")

    def _callback(self, weights: np.ndarray, obj_func_eval: float) -> None:
        """Optimizer callback recording cost function evaluations."""
        elapsed = time.perf_counter() - self._start_time
        self.training_history.add_step(
            loss_val=obj_func_eval,
            weights=weights,
            elapsed=elapsed,
        )
        if len(self.training_history.loss_history) % 10 == 0:
            logger.debug(
                "Step %d: Loss = %.4f (elapsed: %.2fs)",
                len(self.training_history.loss_history),
                obj_func_eval,
                elapsed,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConfigurableVQC":
        """Train the VQC model on features X and labels y.

        Args:
            X: Input features of shape (N, num_qubits).
            y: Binary class labels (N,) with values in {0, 1}.

        Returns:
            self
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)

        if X_arr.shape[1] != self.num_qubits:
            raise ValueError(
                f"Feature dimension {X_arr.shape[1]} does not match model qubits {self.num_qubits}"
            )

        # Initialize weights with fixed seed
        rng = np.random.default_rng(self.random_seed)
        initial_point = rng.uniform(
            low=-np.pi,
            high=np.pi,
            size=self.ansatz.num_parameters,
        )

        logger.info(
            "Training VQC: %d samples, %d qubits, %d layers, %d trainable parameters...",
            len(X_arr),
            self.num_qubits,
            self.num_layers,
            self.ansatz.num_parameters,
        )

        # Build QML VQC
        self.vqc_internal = VQC(
            num_qubits=self.num_qubits,
            feature_map=self.feature_map.circuit,
            ansatz=self.ansatz.circuit,
            loss="cross_entropy",
            optimizer=self.optimizer,
            warm_start=False,
            initial_point=initial_point,
            callback=self._callback,
            sampler=self.sampler,
            interpret=self.interpret_func,
            output_shape=2,  # Binary classification
        )

        self._start_time = time.perf_counter()
        self.vqc_internal.fit(X_arr, y_arr)
        total_time = time.perf_counter() - self._start_time

        # Record fit results and loss trajectory
        fit_res = getattr(self.vqc_internal, "_fit_result", None)
        if fit_res is not None:
            n_evals = getattr(fit_res, "nfev", 0) or getattr(fit_res, "nit", 0) or 1
            fun_val = getattr(fit_res, "fun", None)
            best_x = getattr(fit_res, "x", None)

            if len(self.training_history.loss_history) == 0 and fun_val is not None:
                self.training_history.add_step(
                    loss_val=float(fun_val),
                    weights=best_x,
                    elapsed=total_time,
                )
            if self.training_history.total_evaluations == 0 or self.training_history.total_evaluations < n_evals:
                self.training_history.total_evaluations = int(n_evals)

        self.training_history.training_time_seconds = total_time
        self.training_history.converged = True
        self.is_fitted = True

        logger.info(
            "VQC training completed in %.2f seconds (%d evaluations, final loss=%.4f)",
            total_time,
            self.training_history.total_evaluations,
            self.training_history.best_loss,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Args:
            X: Array of shape (N, num_qubits).

        Returns:
            1D array of predicted class labels {0, 1}.
        """
        if not self.is_fitted or self.vqc_internal is None:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=float)
        preds = self.vqc_internal.predict(X_arr)
        preds_arr = np.asarray(preds, dtype=int)
        if preds_arr.ndim > 1 and preds_arr.shape[1] > 1:
            return np.argmax(preds_arr, axis=1)
        return preds_arr.ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Args:
            X: Array of shape (N, num_qubits).

        Returns:
            2D array of probabilities (N, 2).
        """
        if not self.is_fitted or self.vqc_internal is None:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=float)
        probas = self.vqc_internal.predict_proba(X_arr)
        return np.asarray(probas, dtype=float)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on given test data and labels."""
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y, dtype=int)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Evaluate VQC model and return standardized ClassificationReport."""
        from src.evaluation.classification_metrics import compute_classification_metrics

        preds = self.predict(X)
        probas = self.predict_proba(X)
        return compute_classification_metrics(np.asarray(y, dtype=int), preds, probas)

    @property
    def weights(self) -> np.ndarray | None:
        """Trained parameter weights."""
        if self.vqc_internal is not None and hasattr(self.vqc_internal, "weights"):
            return self.vqc_internal.weights
        return None

    def save(self, path: str | Path) -> None:
        """Save model checkpoint to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "rotation_gates": self.rotation_gates,
            "entanglement_type": self.entanglement_type,
            "entanglement_freq": self.entanglement_freq,
            "feature_rotation": self.feature_rotation,
            "optimizer_name": self.optimizer_name,
            "maxiter": self.maxiter,
            "random_seed": self.random_seed,
            "measurement_strategy": self.measurement_strategy,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "metadata": self.metadata.to_dict(),
            "training_history": self.training_history.to_dict(),
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        logger.info("Saved VQC checkpoint to '%s'", path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> dict[str, Any]:
        """Load model checkpoint data from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load(cls, path: str | Path) -> "ConfigurableVQC":
        """Load and reconstruct a trained VQC model from disk without retraining."""
        checkpoint = cls.load_checkpoint(path)
        model = cls(
            num_qubits=checkpoint["num_qubits"],
            num_layers=checkpoint["num_layers"],
            rotation_gates=checkpoint["rotation_gates"],
            entanglement_type=checkpoint["entanglement_type"],
            entanglement_freq=checkpoint["entanglement_freq"],
            feature_rotation=checkpoint["feature_rotation"],
            measurement_strategy=checkpoint["measurement_strategy"],
            optimizer_name=checkpoint.get("optimizer_name", "cobyla"),
            maxiter=checkpoint.get("maxiter", 35),
            random_seed=checkpoint.get("random_seed", 42),
        )
        if checkpoint.get("weights") is not None:
            from qiskit_algorithms.optimizers import OptimizerResult

            model.vqc_internal = VQC(
                num_qubits=model.num_qubits,
                feature_map=model.feature_map.circuit,
                ansatz=model.ansatz.circuit,
                loss="cross_entropy",
                optimizer=model.optimizer,
                sampler=model.sampler,
                interpret=model.interpret_func,
                output_shape=2,
            )
            res = OptimizerResult()
            res.x = np.array(checkpoint["weights"], dtype=float)
            model.vqc_internal._fit_result = res
            model.is_fitted = True
        return model
