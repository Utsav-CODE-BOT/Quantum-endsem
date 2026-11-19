# System Architecture: Self-Adaptive Quantum Neural Networks

## 1. System Overview

The **Self-Adaptive Quantum Neural Network with Resource-Aware Architecture Optimization** framework is an automated, modular research software platform designed to search for and optimize Variational Quantum Classifier (VQC) architectures.

The primary research objective is to resolve the trade-off between:
1. **Predictive Performance:** Classification accuracy, precision, recall, F1-score, and ROC-AUC on low-dimensional image representations.
2. **Quantum Resource Overhead:** Circuit depth, total gate count, two-qubit entangling (CNOT) gate count, and parameter dimension.
3. **Hardware Noise Robustness:** Degradation tolerance under realistic depolarizing and readout noise channels in Noisy Intermediate-Scale Quantum (NISQ) environments.

---

## 2. Global Pipeline Architecture

```
                                  [Raw Fashion-MNIST]
                                           │
                                           ▼
                                 [Dataset Splitter]
                           (Train: 200, Val: 60, Test: 100)
                                           │
                                           ▼
                                   [PCA Reducer]
                              (Strict Fit on Train ONLY)
                                           │
                                           ▼
                                   [Feature Scaler]
                                (Normalized to [0, π])
                                           │
                                           ▼
                               [Quantum Angle Feature Map]
                                (Ry rotations on 4 Qubits)
                                           │
                                           ▼
   ┌─────────────────────────────── Evolutionary Architecture Loop ─────────────────────────────┐
   │                                                                                            │
   │   [Population Generator] ──► [Ansatz Circuit Generator] ──► [Configurable VQC (COBYLA)]    │
   │            ▲                                                            │                  │
   │            │                                                            ▼                  │
   │     [Genetic Operators]                                     [Validation Evaluator]         │
   │   (Crossover, Mutation,    ◄── [Fitness Engine] ◄────────── (Accuracy, Depth, Gates)       │
   │     Elitist Selection)      (w_p·Perf - w_c·Comp + w_r·Rob)                                │
   │                                                                                            │
   └─────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                             │
                                             ▼ (Best Architecture)
                             [Noise-Aware Hardware Evaluation]
                       (Depolarizing + Readout Noise: AerSimulator)
                                             │
                                             ▼
                               [Multi-Model Benchmarking]
                        (Linear SVM, RBF SVM, MLP, Fixed VQCs)
                                             │
                                             ▼
                             [Figures & Research Reporting]
```

---

## 3. Modular Component Breakdown

The codebase is strictly modularized across clear functional boundaries:

### 3.1 Data Ingestion & Preprocessing (`src/data`, `src/preprocessing`)
- **`dataset_loader.py`:** Downloads and extracts Fashion-MNIST binary classes (Class 0: T-shirt vs Class 9: Ankle boot).
- **`dataset_splitter.py`:** Performs stratified splitting into independent Train (200), Validation (60), and Test (100) partitions.
- **`pca_reducer.py`:** Fits Principal Component Analysis (PCA) exclusively on the training split to extract the top $d=4$ principal components. Validation and test sets are strictly transformed using the fitted training projection, preventing data leakage.
- **`feature_scaler.py`:** Rescales features to $[0, \pi]$ suitable for lossless single-qubit quantum rotation encoding.

### 3.2 Quantum Circuit Engineering (`src/quantum`)
- **`feature_encoding.py`:** `AngleFeatureMap` encoding feature vectors $\mathbf{x} \in [0, \pi]^4$ via parameterized rotation gates ($R_y(x_i)$).
- **`ansatz_builder.py`:** `AnsatzBuilder` constructing parameterized variational layers with configurable rotation gates ($R_x, R_y, R_z$), entanglement topologies (`linear`, `circular`, `pairwise`, `none`), and layer repetition counts.
- **`circuit_generator.py`:** `VQCCircuitGenerator` compiling the full quantum circuit and extracting structural metadata (circuit depth, gate distribution, parameter count, CX count).
- **`measurement.py`:** Bitstring interpretation strategies mapping multi-qubit measurement outcomes $\sum b_i \pmod 2 \to \{0, 1\}$.
- **`simulator.py`:** `QuantumSimulatorManager` encapsulating statevector ideal simulation and noisy Qiskit Aer backends.
- **`vqc_model.py`:** `ConfigurableVQC` adhering to scikit-learn interfaces (`fit`, `predict`, `predict_proba`, `score`, `evaluate`) with COBYLA classical optimization.

### 3.3 Evolutionary Architecture Optimization (`src/optimization`, `src/evaluation`)
- **`architecture_space.py`:** Defines the discrete gene space and structural validity constraints for candidate quantum circuits.
- **`architecture_generator.py`:** Population sampling and chromosome generation.
- **`fitness.py`:** Normalized multi-objective scalar fitness formulation:
  $$F(A) = w_1 \cdot \text{Acc}_{\text{val}}(A) - w_2 \cdot \mathcal{C}(A) + w_3 \cdot \mathcal{R}(A)$$
- **`evaluator.py`:** `ArchitectureEvaluator` with evaluation caching and validation scoring.
- **`selection.py`, `crossover.py`, `mutation.py`:** Tournament selection, uniform crossover, and structural mutation operators.
- **`evolutionary_search.py`:** `EvolutionarySearchEngine` orchestrating generations, elitism preservation, population tracking, and early stopping.

### 3.4 Hardware Noise Evaluation & Benchmarking (`src/evaluation`, `src/baselines`)
- **`noise_evaluator.py`:** Constructs parameterized Qiskit Aer noise models (single-qubit depolarizing, two-qubit depolarizing, measurement bit-flip) and evaluates accuracy degradation.
- **`classical_svm.py` & `classical_nn.py`:** Scikit-learn Linear SVM, RBF SVM, and MLP baselines for classical-quantum parity comparisons.
- **`fixed_vqc.py`:** Canonical fixed shallow (1-layer), standard (2-layer), and deep (4-layer) VQC baselines.
