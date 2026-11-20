"""Noise simulation engine for evaluating VQC robustness under realistic quantum errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
)
from qiskit_aer.primitives import Sampler as AerSampler

from src.evaluation.classification_metrics import (
    ClassificationReport,
    compute_classification_metrics,
)
from src.quantum.vqc_model import ConfigurableVQC
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NoiseModelConfig:
    """Configurable quantum hardware noise parameters."""

    name: str = "custom_depolarizing"
    single_qubit_error: float = 0.001  # Depolarizing error for 1-qubit gates (ry, rx, rz)
    two_qubit_error: float = 0.01  # Depolarizing error for 2-qubit gates (cx)
    measurement_error: float = 0.01  # Probability of bit-flip during measurement
    shots: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NoiseEvaluationReport:
    """Quantitative comparison between ideal and noisy quantum evaluation."""

    arch_id: str
    noise_config: dict[str, Any]
    ideal_metrics: ClassificationReport
    noisy_metrics: ClassificationReport
    accuracy_drop: float  # ideal_acc - noisy_acc
    f1_drop: float  # ideal_f1 - noisy_f1
    relative_accuracy_retention: float  # noisy_acc / max(ideal_acc, 1e-6)
    noise_sensitivity_index: float  # relative drop
    noisy_probabilities: np.ndarray | None = None
    ideal_probabilities: np.ndarray | None = None
    noisy_predictions: np.ndarray | None = None
    ideal_predictions: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "noise_config": self.noise_config,
            "ideal_metrics": self.ideal_metrics.to_dict(),
            "noisy_metrics": self.noisy_metrics.to_dict(),
            "accuracy_drop": round(self.accuracy_drop, 4),
            "f1_drop": round(self.f1_drop, 4),
            "relative_accuracy_retention": round(self.relative_accuracy_retention, 4),
            "noise_sensitivity_index": round(self.noise_sensitivity_index, 4),
        }


class NoiseModelBuilder:
    """Constructs realistic Qiskit Aer noise models with gate and readout errors."""

    @staticmethod
    def build(config: NoiseModelConfig) -> NoiseModel:
        """Construct Qiskit Aer NoiseModel from configuration parameters.

        Args:
            config: NoiseModelConfig dataclass.

        Returns:
            Configured Qiskit Aer NoiseModel.
        """
        noise_model = NoiseModel()

        # 1. Single-qubit depolarizing error for rotation gates
        if config.single_qubit_error > 0.0:
            err_1q = depolarizing_error(config.single_qubit_error, 1)
            noise_model.add_all_qubit_quantum_error(err_1q, ["ry", "rx", "rz", "h"])

        # 2. Two-qubit depolarizing error for entangling CNOT gates
        if config.two_qubit_error > 0.0:
            err_2q = depolarizing_error(config.two_qubit_error, 2)
            noise_model.add_all_qubit_quantum_error(err_2q, ["cx", "cz"])

        # 3. Measurement readout bit-flip error
        if config.measurement_error > 0.0:
            p = config.measurement_error
            # Readout error matrix: [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
            readout_err = ReadoutError([[1.0 - p, p], [p, 1.0 - p]])
            noise_model.add_all_qubit_readout_error(readout_err)

        logger.debug(
            "Constructed NoiseModel '%s': 1q=%.4f, 2q=%.4f, readout=%.4f",
            config.name,
            config.single_qubit_error,
            config.two_qubit_error,
            config.measurement_error,
        )
        return noise_model


class NoiseEvaluator:
    """Evaluates trained VQCs under both ideal and simulated noisy quantum environments."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def evaluate(
        self,
        vqc: ConfigurableVQC,
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_config: NoiseModelConfig,
    ) -> NoiseEvaluationReport:
        """Run paired ideal vs noisy evaluation on the exact same test dataset.

        Args:
            vqc: Fitted ConfigurableVQC model.
            X_test: Test features (N, n_qubits).
            y_test: Test labels (N,).
            noise_config: Noise parameters for simulation.

        Returns:
            NoiseEvaluationReport with degradation metrics.
        """
        if not vqc.is_fitted:
            raise RuntimeError("VQC must be fitted before noise evaluation.")

        X_arr = np.asarray(X_test, dtype=float)
        y_arr = np.asarray(y_test, dtype=int)

        # 1. Ideal prediction (statevector, no noise)
        ideal_preds = vqc.predict(X_arr)
        ideal_probs = vqc.predict_proba(X_arr)
        ideal_metrics = compute_classification_metrics(y_arr, ideal_preds, ideal_probs)

        # 2. Noisy prediction via AerSimulator
        noise_model = NoiseModelBuilder.build(noise_config)
        noisy_sampler = AerSampler(
            backend_options={
                "method": "density_matrix",
                "shots": noise_config.shots,
                "seed_simulator": self.seed,
            },
            run_options={"shots": noise_config.shots},
        )
        noisy_sampler.set_options(noise_model=noise_model)

        # Re-instantiate internal VQC with noisy sampler and pre-trained weights
        from qiskit_machine_learning.algorithms import VQC as QMLVQC
        noisy_vqc_internal = QMLVQC(
            num_qubits=vqc.num_qubits,
            feature_map=vqc.feature_map.circuit,
            ansatz=vqc.ansatz.circuit,
            loss="cross_entropy",
            sampler=noisy_sampler,
            interpret=vqc.interpret_func,
            output_shape=2,
            initial_point=vqc.weights,
        )
        # In QML VQC, to set weights without re-training:
        # Initializing initial_point and fitting with 0 iterations or evaluating predict:
        # Calling fit with maxiter=0 or setting _fit_result:
        from qiskit_algorithms.optimizers import COBYLA
        noisy_vqc_internal._optimizer = COBYLA(maxiter=1)
        # Use warm start trick or fit on 2 samples with 1 iteration, then assign weights
        # Better: predict uses the weights directly if assigned
        noisy_vqc_internal._fit_result = None
        # Call internal predict:
        # In QML VQC, predict calls self._neural_network.forward(X, weights)
        forward_probs = noisy_vqc_internal.neural_network.forward(X_arr, vqc.weights)
        # Normalize forward probabilities and compute predictions
        noisy_probs = np.asarray(forward_probs, dtype=float)
        # Ensure row sums = 1
        row_sums = noisy_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        noisy_probs = noisy_probs / row_sums
        noisy_preds = np.argmax(noisy_probs, axis=1)

        noisy_metrics = compute_classification_metrics(y_arr, noisy_preds, noisy_probs)

        # 3. Compute degradation and sensitivity metrics
        acc_drop = ideal_metrics.accuracy - noisy_metrics.accuracy
        f1_drop = ideal_metrics.f1_macro - noisy_metrics.f1_macro
        retention = noisy_metrics.accuracy / max(ideal_metrics.accuracy, 1e-4)
        sensitivity = max(0.0, acc_drop) / max(ideal_metrics.accuracy, 1e-4)

        self.last_sampler = noisy_sampler
        self.last_noise_model = noise_model

        report = NoiseEvaluationReport(
            arch_id=getattr(vqc, "arch_id", "vqc_model"),
            noise_config=noise_config.to_dict(),
            ideal_metrics=ideal_metrics,
            noisy_metrics=noisy_metrics,
            accuracy_drop=acc_drop,
            f1_drop=f1_drop,
            relative_accuracy_retention=retention,
            noise_sensitivity_index=sensitivity,
            noisy_probabilities=noisy_probs,
            ideal_probabilities=ideal_probs,
            noisy_predictions=noisy_preds,
            ideal_predictions=ideal_preds,
        )

        logger.info(
            "Noise Eval [%s]: Ideal Acc=%.2f%% -> Noisy Acc=%.2f%% (Drop=%.2f%%, Retention=%.2f%%)",
            noise_config.name,
            ideal_metrics.accuracy * 100,
            noisy_metrics.accuracy * 100,
            acc_drop * 100,
            retention * 100,
        )
        return report
