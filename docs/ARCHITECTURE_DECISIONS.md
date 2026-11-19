# Architecture Decision Log

## Project: Self-Adaptive Quantum Neural Networks with Resource-Aware Architecture Optimization

---

### AD-1: Qiskit 1.4.2 over Qiskit 2.6.1

- **Date:** 2026-10-31
- **Context:** Qiskit ecosystem has two active major versions. QML 0.9.1 requires Qiskit >=2.0, while QML 0.8.4 works with Qiskit 1.4.x.
- **Decision:** Use Qiskit 1.4.2 + QML 0.8.4 + Aer 0.16.4 + Algorithms 0.4.0
- **Rationale:** (1) More mature, well-documented VQC API. (2) pip resolution confirmed clean. (3) Extensive community examples. (4) Lower risk of hitting undocumented edge cases.
- **Consequences:** Cannot use features introduced only in Qiskit 2.x. Acceptable for this project's scope.
- **Status:** Approved & Verified

---

### AD-2: Dedicated Isolated Virtual Environment

- **Date:** 2026-10-31
- **Context:** System Python has numpy 2.4.6 and scipy 1.17.1. Qiskit stack requires scipy <1.16 and may need numpy <2.0.
- **Decision:** Create a dedicated `.venv` inside the project directory.
- **Rationale:** Prevents dependency conflicts with the user's existing Python environment.
- **Consequences:** All project commands must be run inside the activated venv.
- **Status:** Approved & Verified

---

### AD-3: Sampler-Based VQC as Primary Baseline with Primitive Clarification

- **Date:** 2026-10-31 (Updated following user Phase 0 clarification)
- **Context:** In Qiskit Machine Learning 0.8.4, `VQC` natively wraps `SamplerQNN` and uses a `BaseSampler` primitive (defaulting to `qiskit.primitives.Sampler`). It classifies by measuring bitstring probabilities and transforming them via an `interpret` function (parity mapping for binary classification). Expectation-value-based classification uses `EstimatorQNN` with `StatevectorEstimator`.
- **Decision:** Use standard Qiskit Machine Learning `VQC` with the compatible `Sampler` primitive for the primary baseline and candidate evaluations. Distinguish this clearly from custom `EstimatorQNN` expectation-value architectures.
- **Rationale:** Meets user guidance to prefer the standard tested QML VQC baseline while maintaining clean separation of concerns and exact reproducibility.
- **Consequences:** The baseline VQC uses probability distribution over computational basis states with parity post-processing.
- **Status:** Approved & Verified

---

### AD-4: Angle Encoding (RY) as Initial Feature Map

- **Date:** 2026-10-31
- **Context:** Multiple quantum feature encoding strategies exist (angle, amplitude, IQP, ZZ).
- **Decision:** Start with simple RY angle encoding: one qubit per PCA feature.
- **Rationale:** Simple, well-understood, direct mapping. n_qubits = n_features (4, 6, or 8).
- **Consequences:** Limits expressivity compared to entangling feature maps. Can be extended later.
- **Status:** Approved & Verified

---

### AD-5: Binary Classification First

- **Date:** 2026-10-31
- **Context:** Fashion-MNIST has 10 classes. VQC output mapping for multiclass requires multiple measurement strategies.
- **Decision:** Implement binary classification first (Class 0: T-shirt vs Class 9: Ankle boot).
- **Rationale:** Simpler VQC output mapping (single expectation value / bitstring parity → threshold → class). Faster to validate. Reduces debugging surface.
- **Consequences:** Initial results apply to selected class pairs with high visual contrast.
- **Status:** Approved & Verified

---

### AD-6: COBYLA as Default Classical Optimizer

- **Date:** 2026-10-31
- **Context:** Classical optimizers for VQC training include gradient-based (L-BFGS-B, Adam) and gradient-free (COBYLA, SPSA, Nelder-Mead).
- **Decision:** Default to COBYLA (35 iterations per candidate).
- **Rationale:** Derivative-free, works robustly with noisy quantum cost landscapes, avoids parameter shift gradient evaluation overhead.
- **Consequences:** Convergence rate is predictable and fast on small parameter spaces.
- **Status:** Approved & Verified

---

### AD-7: Paired Ideal vs Noisy Evaluation via Density Matrix AerSimulator

- **Date:** 2026-10-31
- **Context:** Evaluating quantum hardware noise requires simulating non-unitary channels (depolarizing and readout error).
- **Decision:** Implement `NoiseEvaluator` using `qiskit_aer.AerSimulator` with `method="density_matrix"` and 1,024 shots. Compare paired ideal statevector and noisy predictions on the exact same test dataset.
- **Rationale:** Density matrix simulation faithfully tracks mixed states under depolarizing noise. Comparing the identical weights under ideal and noisy conditions isolates hardware error susceptibility from parameter optimization quality.
- **Consequences:** Simulates 4 qubits with full density matrices ($16 \times 16 = 256$ complex entries) rapidly in ~0.5s per candidate.
- **Status:** Approved & Verified

---

### AD-8: Normalized Multi-Objective Scalar Fitness Formulation

- **Date:** 2026-10-31
- **Context:** Architecture search must balance classification performance, quantum circuit complexity (depth and gate count), and simulated noise robustness without falling into local minima.
- **Decision:** Define scalar fitness as:
  $$F(A) = 0.50 \cdot \text{Acc}_{\text{val}}(A) - 0.30 \cdot \text{Complexity}(A) + 0.20 \cdot \text{Robustness}(A)$$
  where $\text{Complexity}(A) = \frac{1}{2}\left(\frac{\text{depth}}{\text{depth}_{\max}} + \frac{N_{2Q}}{\text{CX}_{\max}}\right)$ with $\text{depth}_{\max} = 50$ and $\text{CX}_{\max} = 30$.
- **Rationale:** Prevents the optimizer from favoring arbitrarily deep circuits that overfit or fail under noise. Directly rewards compact NISQ-friendly topologies by penalizing error-prone two-qubit entangling gates.
- **Consequences:** Autonomously steered the search toward a depth-5 circuit with 16 parameters achieving 96% accuracy.
- **Status:** Approved & Verified

---

### AD-9: Classical Baseline Selection for Fair Fair-Feature Benchmarking

- **Date:** 2026-10-31
- **Context:** Comparing quantum classifiers against classical models must be conducted on identical preprocessed feature representations to prevent bias.
- **Decision:** Implement Linear SVM, RBF SVM, and a compact Classical MLP (25 trainable weights, directly matching the 16-24 quantum parameter budget) evaluated on the identical 4 PCA features.
- **Rationale:** Provides rigorous empirical context without fabricating quantum superiority. Linear and RBF SVMs achieve 100% accuracy in milliseconds, demonstrating that classical methods excel on low dimensions while highlighting that the quantum framework's contribution is automated circuit resource reduction.
- **Consequences:** Establishes clear, honest scientific benchmarking.
- **Status:** Approved & Verified

---

### AD-10: Decoupling VQC Evaluation from Evaluation Package

- **Date:** 2026-10-31
- **Context:** A circular import occurred between `src.quantum.vqc_model` and `src.evaluation.evaluator` when `vqc_model.py` imported `classification_metrics.py` at the module level.
- **Decision:** Decouple `vqc_model.evaluate` by performing a lazy import of `compute_classification_metrics` inside the method.
- **Rationale:** Resolves module loading order conflicts and maintains a strict unidirectional dependency graph (`quantum` → `training` → `evaluation` → `optimization`).
- **Consequences:** All 92 unit and integration tests collect and run cleanly without circular dependency errors.
- **Status:** Approved & Verified
