"""Validation of quantum primitives, Qiskit 1.4.2, and Qiskit Machine Learning 0.8.4.

Confirms that:
1. Basic circuits and parameter assignment function correctly.
2. Sampler-based VQC in Qiskit Machine Learning runs without issue.
3. Estimator-based classification (expectation-value approach) is explicitly distinguished.
4. AerSimulator and noise models can be instantiated and executed.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import Sampler, StatevectorEstimator, StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN


class TestQuantumCircuits:
    """Verify core Qiskit 1.4.2 circuit operations."""

    def test_parametric_circuit_creation(self) -> None:
        theta = ParameterVector("θ", 4)
        qc = QuantumCircuit(2)
        qc.ry(theta[0], 0)
        qc.ry(theta[1], 1)
        qc.cx(0, 1)
        qc.ry(theta[2], 0)
        qc.ry(theta[3], 1)

        assert qc.num_qubits == 2
        assert qc.num_parameters == 4
        assert qc.depth() > 0

    def test_statevector_sampler_execution(self) -> None:
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        sampler = StatevectorSampler()
        job = sampler.run([qc], shots=100)
        result = job.result()
        pub_result = result[0]
        # In Qiskit 1.4.2 V2 StatevectorSampler, data is accessed via data.meas
        counts = pub_result.data.meas.get_counts()
        assert "00" in counts or "11" in counts
        assert "01" not in counts
        assert "10" not in counts


class TestQMLPrimitivesComparison:
    """Verify SamplerQNN / VQC vs EstimatorQNN and their distinctions."""

    def test_sampler_vqc_training_smoke(self) -> None:
        """Standard QML VQC uses SamplerQNN with parity interpret by default."""
        feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
        ansatz = RealAmplitudes(num_qubits=2, reps=1)
        optimizer = COBYLA(maxiter=10)

        vqc = VQC(
            num_qubits=2,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=Sampler(),
        )

        X = np.array([[0.2, 0.4], [0.8, 0.9]])
        y = np.array([0, 1])

        vqc.fit(X, y)
        preds = vqc.predict(X)
        assert preds.shape == (2,)
        assert all(p in [0, 1] for p in preds)

    def test_estimator_qnn_expectation_value_distinction(self) -> None:
        """Verify that expectation-value (EstimatorQNN) can be constructed for comparison."""
        from qiskit.quantum_info import SparsePauliOp

        qc = QuantumCircuit(2)
        x = ParameterVector("x", 2)
        w = ParameterVector("w", 2)
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)
        qc.cx(0, 1)
        qc.ry(w[0], 0)
        qc.ry(w[1], 1)

        observable = SparsePauliOp.from_list([("ZZ", 1.0)])
        estimator = StatevectorEstimator()

        qnn = EstimatorQNN(
            circuit=qc,
            observables=[observable],
            input_params=x,
            weight_params=w,
            estimator=estimator,
        )

        # Forward pass evaluation
        input_data = np.array([[0.1, 0.2]])
        weights = np.array([0.5, -0.5])
        forward_val = qnn.forward(input_data, weights)
        assert forward_val.shape == (1, 1)
        # Expectation value of ZZ must lie in [-1, 1]
        assert -1.0 <= forward_val[0, 0] <= 1.0


class TestAerNoiseModel:
    """Verify AerSimulator and noise model instantiation."""

    def test_depolarizing_noise_simulation(self) -> None:
        noise_model = NoiseModel()
        error_1q = depolarizing_error(0.05, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ["ry", "rx"])

        sim = AerSimulator(noise_model=noise_model)
        qc = QuantumCircuit(1)
        qc.ry(np.pi / 2, 0)
        qc.measure_all()

        job = sim.run(qc, shots=500)
        result = job.result()
        counts = result.get_counts()
        assert sum(counts.values()) == 500
