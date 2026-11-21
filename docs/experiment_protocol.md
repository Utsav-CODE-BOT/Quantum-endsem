# Experimental Protocol & Reproducibility Guidelines

## 1. Dataset Selection & Partitioning

### 1.1 Dataset Specification
- **Dataset:** Fashion-MNIST (Zalando Research)
- **Binary Classification Subproblem:**
  - Class 0: T-shirt / top
  - Class 9: Ankle boot
  - Justification: These classes provide clean visual contrast between upper-body apparel and footwear, making them an ideal benchmark for low-dimensional quantum encoding without introducing extreme non-separability.

### 1.2 Partitioning & Stratification
- **Training Set ($N_{\text{train}} = 200$):** 100 samples of Class 0, 100 samples of Class 9.
- **Validation Set ($N_{\text{val}} = 60$):** 30 samples of Class 0, 30 samples of Class 9. Used strictly for candidate evaluation and fitness calculation during architecture search.
- **Test Set ($N_{\text{test}} = 100$):** 50 samples of Class 0, 50 samples of Class 9. Held out strictly for final reporting and noise robustness benchmarking; never exposed to architecture selection.

---

## 2. Zero-Leakage Preprocessing Protocol

To adhere to rigorous scientific machine learning standards, data leakage is strictly prevented:

```
[All Raw Images] ──► [Split into Train / Val / Test]
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      [Train Split ONLY]          [Val & Test Splits]
               │                         │
               ▼                         │
        Fit PCA (k=4)                    │
        Fit MinMax Scaler                │
               │                         │
               ├─────────────────────────┼──► Transform Val with Train PCA & Scaler
               │                         │
               ▼                         ▼
       Transform Train            Transform Test
```

1. **Feature Extraction:** PCA is fitted **only on the training split** ($\mathbf{X}_{\text{train}}$).
2. **Feature Projection:** Validation ($\mathbf{X}_{\text{val}}$) and test ($\mathbf{X}_{\text{test}}$) samples are projected using the pre-fitted training projection matrix $\mathbf{W}_{\text{PCA}}$.
3. **Angle Scaling:** Feature min/max values are computed strictly from the training projections and used to scale all splits into $[0, \pi]$.
4. **Validation:** Automated unit tests in `tests/test_preprocessing.py` verify that mutating test data does not alter PCA components or scaler statistics.

---

## 3. Quantum Hardware Noise Simulation Protocol

Simulated noise experiments utilize `qiskit_aer.AerSimulator` configured with the `density_matrix` simulation method and 1,024 shots. Three standard NISQ noise regimes are benchmarked:

| Noise Regime | 1-Qubit Gate Error ($p_{1q}$) | 2-Qubit Gate Error ($p_{2q}$) | Readout Bit-Flip Error ($p_{\text{meas}}$) | Simulated Backend |
|:---|:---:|:---:|:---:|:---:|
| **Ideal** | $0.000$ | $0.000$ | $0.000$ | Statevector Sampler |
| **Low Noise** | $0.0005$ ($0.05\%$) | $0.005$ ($0.5\%$) | $0.005$ ($0.5\%$) | Aer Density Matrix (1024 shots) |
| **Medium Noise** | $0.0010$ ($0.10\%$) | $0.010$ ($1.0\%$) | $0.015$ ($1.5\%$) | Aer Density Matrix (1024 shots) |
| **High Noise** | $0.0050$ ($0.50\%$) | $0.030$ ($3.0\%$) | $0.030$ ($3.0\%$) | Aer Density Matrix (1024 shots) |

- **Error Channels:**
  - Single-qubit depolarizing error applied to $\{R_x, R_y, R_z, H\}$ gates.
  - Two-qubit depolarizing error applied to $\{\text{CX}, \text{CZ}\}$ gates.
  - Readout error matrix $\begin{pmatrix} 1-p & p \\ p & 1-p \end{pmatrix}$ applied to computational basis measurements.

---

## 4. Reproducibility & Determinism

All stochastic operations are strictly governed by explicit pseudo-random seeds:
- Dataset shuffling and partitioning: `seed = 42`
- PCA decomposition: `seed = 42`
- Initial quantum parameter sampling: `seed = 42`
- Qiskit Aer simulator: `seed_simulator = 42`
- Evolutionary genetic operators (selection, crossover, mutation): `seed = 42`

All hardware specifications, OS parameters, package versions, and timestamps are logged systematically in `experiments/logs/`.

---

## 5. Optimization Budget Specification & Multi-Objective Scalarization

### 5.1 Multi-Objective Scalarization
The optimization engine uses a normalized weighted-sum scalarization objective:
$$F(A) = w_{\text{perf}} \cdot f_{\text{perf}}(A) - w_{\text{comp}} \cdot f_{\text{comp}}(A) + w_{\text{rob}} \cdot f_{\text{rob}}(A)$$
with weights $w_{\text{perf}} = 0.50$, $w_{\text{comp}} = 0.30$, and $w_{\text{rob}} = 0.20$.
This formulation guides the single-objective evolutionary search over multiple competing physical and statistical criteria. It does not employ Pareto-dominance sorting algorithms (such as NSGA-II); visual trade-off curves illustrate empirical distributions of evaluated candidates across complexity and accuracy dimensions.

### 5.2 Optimizer Iteration Budgets
Classical COBYLA training iteration budgets are assigned systematically to balance search exploration against computational feasibility:
1. **Evolutionary Search Discovery Budget (`maxiter = 25`):**
   - Applied during candidate evaluation in the genetic algorithm.
   - Run ID: `EXP-SEARCH-4169EE9360` (Discovered `arch_4169ee9360`: 90.00% validation accuracy, 88.00% test accuracy, 16.95s training time).
2. **Standardized Comparative Benchmark Budget (`maxiter = 35`):**
   - Applied uniformly to all fixed classical and quantum architectures in `experiments/results/benchmark_comparison.json`.
   - Run ID: `EXP-BENCH-20261120` (`arch_4169ee9360`: 88.33% validation accuracy, 90.00% test accuracy; Baseline VQC: 98.33% validation accuracy, 95.00% test accuracy).
3. **Exploratory Baseline Budget (`maxiter = 100`):**
   - Evaluated during preliminary baseline convergence profiling (Baseline VQC: 100.00% validation accuracy, 96.00% test accuracy).
